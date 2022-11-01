import yaml
import glob
import logging
import math
import os
import random
import shutil
import time
import cv2
import hashlib
import multiprocessing
import numpy as np
from itertools import repeat
from PIL import Image, ExifTags
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import mindspore as ms
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.communication.management import init, get_rank, get_group_size
# from mindspore.amp import DynamicLossScaler, all_finite
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.ops import operations as P

from network.yolo import Model
from network.common import ModelEMA
from network.loss import *
from config.args import get_args_train
from utils.optimizer import get_group_param_yolov7, get_lr_yolov7
from utils.dataset import create_dataloader
from utils.general import increment_path, colorstr, labels_to_class_weights, check_file, check_img_size, all_finite_cpu

def train(hyp, opt):
    ms.set_seed(2)
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = data_dict["dataset_name"] == "coco"
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Model
    sync_bn = opt.sync_bn and context.get_context("device_target") == "Ascend" and rank_size > 1
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn, opt=opt)  # create
    if rank % 8 == 0:
        ema_model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), sync_bn=sync_bn)
        ema = ModelEMA(ema_model)
    else:
        ema = None

    pretrained = weights.endswith('.ckpt')
    if pretrained:
        param_dict = ms.load_checkpoint(weights)
        ms.load_param_into_net(model, param_dict)
        print(f"Pretrain model load from {weights}")
        if ema and opt.ema_weight.endswith('.ckpt'):
            param_dict_ema = ms.load_checkpoint(opt.ema_weight)
            ms.load_param_into_net(ema.ema_model, param_dict_ema)
            ema.updates = param_dict_ema["updates"]
            print(f"Ema model load from {opt.ema_weight}")

    # Freeze
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(model.stride.asnumpy().max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    train_path = data_dict['train']
    test_path = data_dict['val']
    dataloader, dataset, per_epoch_size = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                                            hyp=hyp, augment=True, cache=opt.cache_images,
                                                            rect=opt.rect, rank_size=opt.rank_size, rank=opt.rank,
                                                            num_parallel_workers=8 if rank_size > 1 else 16,
                                                            image_weights=opt.image_weights, quad=opt.quad,
                                                            prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = per_epoch_size  # number of batches per epoch
    assert mlc < nc, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, nc, opt.data, nc - 1)

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")
    if "yolov7" in opt.cfg:
        pg0, pg1, pg2 = get_group_param_yolov7(model)
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov7(opt, hyp, per_epoch_size)
        assert len(momentum_pg) == warmup_steps
        momentum_pg = Tensor(momentum_pg, ms.float32)
    else:
        raise NotImplementedError
    group_params = [{'params': pg0, 'lr': lr_pg0},
                    {'params': pg1, 'lr': lr_pg1, 'weight_decay': hyp['weight_decay']},
                    {'params': pg2, 'lr': lr_pg2}]

    print(f"optimizer loss scale is {opt.ms_optim_loss_scale}")
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True,
                           loss_scale=opt.ms_optim_loss_scale)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999,
                            loss_scale=opt.ms_optim_loss_scale)
    else:
        raise NotImplementedError

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = Tensor(labels_to_class_weights(dataset.labels, nc) * nc)  # attach class weights
    model.names = names
    if ema:
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'gr', 'names', 'stride', 'class_weights'])


    if opt.ms_strategy == "StaticCell":
        train_step = create_train_static_shape_cell(model, optimizer, amp_level=opt.ms_amp_level)
        # train_step = create_train_static_shape_cell_clip_grad(model, optimizer, amp_level=opt.ms_amp_level)
    else:
        # amp
        ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
        if opt.is_distributed:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        else:
            grad_reducer = ops.functional.identity

        if opt.ms_loss_scaler == "dynamic":
            from mindspore.amp import DynamicLossScaler
            loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)
        elif opt.ms_loss_scaler == "static":
            from mindspore.amp import StaticLossScaler
            loss_scaler = StaticLossScaler(opt.ms_loss_scaler_value)
        elif opt.ms_loss_scaler == "none":
            loss_scaler = None
        else:
            raise NotImplementedError

        if opt.ms_strategy == "StaticShape":
            train_step = create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer,
                                                                    amp_level=opt.ms_amp_level,
                                                                    overflow_still_update=opt.overflow_still_update,
                                                                    sens=opt.ms_grad_sens)
            # if opt.clip_grad:
            #     train_step = create_train_static_shape_fn_clip_grad(model, optimizer, loss_scaler, grad_reducer,
            #                                                         amp_level=opt.ms_amp_level,
            #                                                         overflow_still_update=opt.overflow_still_update)
            # else:
            #     train_step = create_train_static_shape_fn(model, optimizer, loss_scaler, grad_reducer,
            #                                               amp_level=opt.ms_amp_level,
            #                                               overflow_still_update=opt.overflow_still_update)
        elif opt.ms_strategy == "MultiShape":
            raise NotImplementedError
        elif opt.ms_strategy == "DynamicShape":
            assert opt.ms_mode == "pynative", f"The dynamic shape function under static graph is under development. Please " \
                                              f"look forward to the subsequent MS version."
            train_step = create_train_dynamic_shape_fn(model, optimizer, loss_scaler, grad_reducer,
                                                       amp_level=opt.ms_amp_level)
        else:
            raise NotImplementedError


    data_loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    s_time = time.time()
    accumulate_grads = None
    accumulate_cur_step = 0

    model.set_train(True)
    optimizer.set_train(True)

    for i, data in enumerate(data_loader):
        if i < warmup_steps:
            xi = [0, warmup_steps]  # x interp
            accumulate = max(1, np.interp(i, xi, [1, nbs / total_batch_size]).round())
            if opt.optimizer == "sgd":
                optimizer.momentum = momentum_pg[i]
                # print("optimizer.momentum: ", optimizer.momentum.asnumpy())

        cur_epoch = (i // per_epoch_size) + 1
        cur_step = (i % per_epoch_size) + 1
        imgs, labels, paths = data["img"], data["label_out"], data["img_files"]
        input_dtype = ms.float32 if opt.ms_amp_level == "O0" else ms.float16
        imgs, labels = Tensor(imgs, input_dtype), Tensor(labels, input_dtype)

        # Multi-scale
        ns = None
        if opt.multi_scale:
            sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                # imgs = ops.interpolate(imgs, sizes=ns, coordinate_transformation_mode="asymmetric", mode="bilinear")

        # Accumulate Grad
        s_train_time = time.time()
        if opt.ms_strategy == "StaticCell":
            assert accumulate == 1, "Grad Accumulate must be 1 when train with StaticCell."
            # 1. TrainOneStepCell or
            loss = train_step(imgs, labels, ns)

            # # 2. TrainOneStepWithLossScaleCell
            # loss, overflow, loss_scale = train_step(imgs, labels, ns)
            # if overflow:
            #     print(f"Epoch: {cur_epoch}, Step: {cur_step}, this step grad overflow, drop. "
            #           f"Cur loss scale is {loss_scale.asnumpy()}")
        else:
            if accumulate == 1:
                _, loss_item, _, grads_finite = train_step(imgs, labels, ns, True)
                if loss_scaler:
                    loss_scaler.adjust(grads_finite)
                    if not grads_finite:
                        print("overflow, loss scale adjust to ", loss_scaler.scale_value.asnumpy())
            else:
                _, loss_item, grads, grads_finite = train_step(imgs, labels, ns, False)
                if loss_scaler:
                    loss_scaler.adjust(grads_finite)
                if grads_finite:
                    accumulate_cur_step += 1
                    if accumulate_grads:
                        assert len(accumulate_grads) == len(grads)
                        for gi in range(len(grads)):
                            accumulate_grads[gi] += grads[gi]
                    else:
                        accumulate_grads = list(grads)

                    if accumulate_cur_step % accumulate == 0:
                        optimizer(tuple(accumulate_grads))
                        print(f"-Epoch: {cur_epoch}, Step: {cur_step}, optimizer an accumulate step success.")
                        # reset accumulate
                        accumulate_grads = None
                        accumulate_cur_step = 0
                else:
                    if loss_scaler:
                        print(f"Epoch: {cur_epoch}, Step: {cur_step}, this step grad overflow, drop. "
                              f"Loss scale adjust to {loss_scaler.scale_value.asnumpy()}")
                    else:
                        print(f"Epoch: {cur_epoch}, Step: {cur_step}, this step grad overflow, drop. ")

        _p_train_size = ns if ns else imgs.shape[2:]
        print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
              f"fp/bp time cost: {(time.time() - s_train_time) * 1000:.2f} ms")
        if opt.ms_strategy == "StaticCell":
            print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
                  f"loss: {loss.asnumpy():.4f}, step time: {(time.time() - s_time) * 1000:.2f} ms")
        else:
            print(f"Epoch {epochs}/{cur_epoch}, Step {per_epoch_size}/{cur_step}, size {_p_train_size}, "
                  f"loss: {loss_item[3].asnumpy():.4f}, lbox: {loss_item[0].asnumpy():.4f}, lobj: "
                  f"{loss_item[1].asnumpy():.4f}, lcls: {loss_item[2].asnumpy():.4f}, "
                  f"step time: {(time.time() - s_time) * 1000:.2f} ms")
        s_time = time.time()

        if (rank % 8 == 0) and ((i + 1) % per_epoch_size == 0):
            # Save Checkpoint
            model_name = os.path.basename(opt.cfg)[:-5] # delete ".yaml"
            ckpt_path = os.path.join(wdir, f"{model_name}_{cur_epoch}.ckpt")
            ms.save_checkpoint(model, ckpt_path)
            if ema:
                ema_ckpt_path = os.path.join(wdir, f"EMA_{model_name}_{cur_epoch}.ckpt")
                ms.save_checkpoint(ema.ema_model, ema_ckpt_path, append_dict={"updates": ema.updates})

        # TODO: eval every epoch

# check code
def check_train(hyp, opt):
    ms.set_seed(2)
    save_dir, epochs, batch_size, total_batch_size, weights, rank, freeze = \
        opt.save_dir, opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, opt.rank, opt.freeze

    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    is_coco = data_dict["dataset_name"] == "coco"
    nc = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Directories
    wdir = os.path.join(save_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    # Save run settings
    with open(os.path.join(save_dir, "hyp.yaml"), 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(os.path.join(save_dir, "opt.yaml"), 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Model
    model = Model(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), opt=opt)  # create
    pretrained = weights.endswith('.ckpt')
    if pretrained:
        param_dict = ms.load_checkpoint(weights)
        ms.load_param_into_net(model, param_dict)

    # Freeze
    freeze = [f'model.{x}.' for x in
              (freeze if len(freeze) > 1 else range(freeze[0]))]  # parameter names to freeze (full or partial)
    for n, p in model.parameters_and_names():
        p.requires_grad = True  # train all layers
        if any(x in n for x in freeze):
            print('freezing %s' % n)
            p.requires_grad = False

    # Image sizes
    gs = max(int(ops.cast(model.stride, ms.float16).max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / total_batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= total_batch_size * accumulate / nbs  # scale weight_decay
    print(f"Scaled weight_decay = {hyp['weight_decay']}")
    per_epoch_size = 10
    if "yolov7" in opt.cfg:
        pg0, pg1, pg2 = get_group_param_yolov7(model)
        lr_pg0, lr_pg1, lr_pg2, momentum_pg, warmup_steps = get_lr_yolov7(opt, hyp, per_epoch_size)
        assert len(momentum_pg) == warmup_steps
        momentum_pg = Tensor(momentum_pg, ms.float32)
    else:
        raise NotImplementedError
    group_params = [{'params': pg0, 'lr': lr_pg0},
                    {'params': pg1, 'lr': lr_pg1, 'weight_decay': hyp['weight_decay']},
                    {'params': pg2, 'lr': lr_pg2}]
    if opt.optimizer == "sgd":
        optimizer = nn.SGD(group_params, learning_rate=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    elif opt.optimizer == "adam":
        optimizer = nn.Adam(group_params, learning_rate=hyp['lr0'], beta1=hyp['momentum'], beta2=0.999)
    else:
        raise NotImplementedError

    # TODO: EMA

    # TODO: Resume

    # TODO: SyncBatchNorm

    # TODO: eval durning train

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    # model.class_weights = Tensor(labels_to_class_weights(dataset.labels, nc) * nc)  # attach class weights
    model.names = names

    # amp
    if opt.is_distributed:
        mean = context.get_auto_parallel_context("gradients_mean")
        degree = context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity
    ms.amp.auto_mixed_precision(model, amp_level=opt.ms_amp_level)
    from mindspore.amp import DynamicLossScaler
    loss_scaler = DynamicLossScaler(2 ** 12, 2, 1000)

    if opt.ms_strategy == "StaticShape":
        train_step = create_train_static_shape_fn(model, optimizer, loss_scaler, grad_reducer)
    elif opt.ms_strategy == "MultiShape":
        raise NotImplementedError
    elif opt.ms_strategy == "DynamicShape":
        assert opt.ms_mode == "pynative", f"The dynamic shape function under static graph is under development. Please " \
                                          f"look forward to the subsequent MS version."
        train_step = create_train_dynamic_shape_fn(model, optimizer, loss_scaler, grad_reducer)
    else:
        raise NotImplementedError

    s_time = time.time()
    input_dtype = ms.float32 if opt.ms_amp_level == "O0" else ms.float16
    for i in range(50):
        imgs = Tensor(np.load("imgs.npy"), input_dtype)[:batch_size, ...]
        labels = Tensor(np.load("labels.npy"), input_dtype)[:batch_size, ...]
        print("imgs/labels size: ", imgs.shape, labels.shape)
        _, loss_item, _, grad_finite = train_step(imgs, labels, None, True)
        loss_scaler.adjust(grad_finite)
        if not grad_finite:
            print(f"step {i}, overflow, adjust grad to {loss_scaler.scale_value.asnumpy()}")
        else:
            print(f"step {i} train success")
        print(f"step {i}, loss: {loss_item}, cost: {(time.time() - s_time) * 1000.:.2f} ms")
        s_time = time.time()
    print("Train Finish.")


# clip grad define ----------------------------------------------------
clip_grad = ops.MultitypeFuncGraph("clip_grad")
hyper_map = ops.HyperMap()
GRADIENT_CLIP_TYPE = 1 # 0, ClipByValue; 1, ClipByNorm;
GRADIENT_CLIP_VALUE = 10.0
@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
        """
        Clip gradients.

        Inputs:
            clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
            clip_value (float): Specifies how much to clip.
            grad (tuple[Tensor]): Gradients.

        Outputs:
            tuple[Tensor]: clipped gradients.
        """
        if clip_type not in (0, 1):
            return grad
        dt = F.dtype(grad)
        if clip_type == 0:
            new_grad = ops.clip_by_value(grad, F.cast(F.tuple_to_array((-clip_value,)), dt),
                                       F.cast(F.tuple_to_array((clip_value,)), dt))
        else:
            new_grad = nn.ClipByNorm()(grad, F.cast(F.tuple_to_array((clip_value,)), dt))
        return new_grad
# ---------------------------------------------------------------------

# clip grad cell define -----------------------------------------------
class TrainOneStepWithClipGradientCell(nn.Cell):
    '''TrainOneStepWithClipGradientCell'''
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainOneStepWithClipGradientCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.hyper_map = C.HyperMap()
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*inputs, sens)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss
# ---------------------------------------------------------------------


def create_train_static_shape_cell_clip_grad(model, optimizer, amp_level="O2"):
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class

    class Warpper(nn.Cell):
        def __init__(self, model, loss):
            super(Warpper, self).__init__(auto_prefix=False)
            self.model = model
            self.loss = loss

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            if use_ota:
                loss, loss_items = self.loss(pred, label, x)
            else:
                loss, loss_items = self.loss(pred, label)
            return loss

    net_with_loss = Warpper(model, compute_loss)

    # 3. TrainOneStepWithClipGradientCell
    train_network = TrainOneStepWithClipGradientCell(net_with_loss, optimizer, sens=1.0)
    ms.amp.auto_mixed_precision(train_network, amp_level=amp_level)

    train_network.set_train()

    return train_network

def create_train_static_shape_cell(model, optimizer, amp_level="O2"):
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class

    class Warpper(nn.Cell):
        def __init__(self, model, loss):
            super(Warpper, self).__init__(auto_prefix=False)
            self.model = model
            self.loss = loss

        def construct(self, x, label, sizes=None):
            x /= 255.0
            if sizes is not None:
                x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
            pred = self.model(x)
            if use_ota:
                loss, loss_items = self.loss(pred, label, x)
            else:
                loss, loss_items = self.loss(pred, label)
            return loss

    net_with_loss = Warpper(model, compute_loss)

    # 1. TrainOneStepWithLossScaleCell
    # ms.amp.auto_mixed_precision(net_with_loss, amp_level=amp_level)
    # manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=1024.0, scale_factor=2, scale_window=1000)
    # # manager = nn.FixedLossScaleUpdateCell(1024.0)
    # train_network = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=manager)

    # 2. TrainOneStepCell
    train_network = nn.TrainOneStepCell(net_with_loss, optimizer, sens=1.0)
    ms.amp.auto_mixed_precision(train_network, amp_level=amp_level)

    # 3. TrainOneStepWithClipGradientCell
    # train_network = TrainOneStepWithClipGradientCell(net_with_loss, optimizer, sens=1024)
    # ms.amp.auto_mixed_precision(train_network, amp_level=amp_level)

    train_network = train_network.set_train()

    return train_network

def create_train_static_shape_fn_clip_grad(model, optimizer, loss_scaler, grad_reducer=None, amp_level="O2",
                                           overflow_still_update=False):
    from mindspore.amp import all_finite
    if loss_scaler is None:
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(1.0)
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class
    ms.amp.auto_mixed_precision(compute_loss, amp_level=amp_level)

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        if use_ota:
            loss, loss_items = compute_loss(pred, label, x)
        else:
            loss, loss_items = compute_loss(pred, label)
        return loss_scaler.scale(loss), loss_items

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        (loss, loss_items), grads = grad_fn(x, label, sizes)
        grads = grad_reducer(grads)
        unscaled_grads = loss_scaler.unscale(grads)
        unscaled_grads = hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), unscaled_grads)
        grads_finite = all_finite(unscaled_grads)
        # _ = loss_scaler.adjust(grads_finite)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(unscaled_grads))
                    print("overflow, still update.")
                else:
                    print("overflow, drop the step.")

        return loss_scaler.unscale(loss), loss_items, unscaled_grads, grads_finite

    return train_step

def create_train_static_shape_fn(model, optimizer, loss_scaler, grad_reducer=None, amp_level="O2",
                                 overflow_still_update=False):
    from mindspore.amp import all_finite
    if loss_scaler is None:
        from mindspore.amp import StaticLossScaler
        loss_scaler = StaticLossScaler(1.0)
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class
    ms.amp.auto_mixed_precision(compute_loss, amp_level=amp_level)

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        if use_ota:
            loss, loss_items = compute_loss(pred, label, x)
        else:
            loss, loss_items = compute_loss(pred, label)
        return loss_scaler.scale(loss), loss_items

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        (loss, loss_items), grads = grad_fn(x, label, sizes)
        grads = grad_reducer(grads)
        unscaled_grads = loss_scaler.unscale(grads)
        grads_finite = all_finite(unscaled_grads)
        # _ = loss_scaler.adjust(grads_finite)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(unscaled_grads))
                    print("overflow, still update.")
                else:
                    print("overflow, drop the step.")

        return loss_scaler.unscale(loss), loss_items, unscaled_grads, grads_finite

    return train_step

def create_train_static_shape_fn_gradoperation(model, optimizer, loss_scaler, grad_reducer=None, amp_level="O2",
                                               overflow_still_update=False, sens=1.0):
    if loss_scaler is not None:
        print("lossscaler not effective on gradoperation fn.")
    from mindspore.amp import all_finite
    # Def train func
    use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    if use_ota:
        compute_loss = ComputeLossOTA_v2(model)  # init loss class
    else:
        compute_loss = ComputeLoss(model)  # init loss class
    ms.amp.auto_mixed_precision(compute_loss, amp_level=amp_level)

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        if use_ota:
            loss, loss_items = compute_loss(pred, label, x)
        else:
            loss, loss_items = compute_loss(pred, label)
        return loss, ops.stop_gradient(loss_items)

    grad_fn = ops.GradOperation(get_by_list=True, sens_param=True)(forward_func, optimizer.parameters)
    sens_value = sens

    @ms.ms_function
    def train_step(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        sens1, sens2 = ops.fill(loss.dtype, loss.shape, sens_value), \
                       ops.fill(loss_items.dtype, loss_items.shape, sens_value)
        grads = grad_fn(x, label, sizes, (sens1, sens2))
        grads = grad_reducer(grads)
        grads_finite = all_finite(grads)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(grads))
            else:
                if overflow_still_update:
                    loss = ops.depend(loss, optimizer(grads))
                    print("overflow, still update.")
                else:
                    print("overflow, drop the step.")

        return loss, loss_items, grads, grads_finite

    return train_step

def create_train_dynamic_shape_fn(model, optimizer, loss_scaler, grad_reducer=None, amp_level="O2"):
    from mindspore.amp import all_finite
    # Def train func
    # use_ota = 'loss_ota' not in hyp or hyp['loss_ota'] == 1
    use_ota = True
    if use_ota:
        compute_loss = ComputeLossOTA_v1_dynamic(model)  # init loss class
    else:
        raise NotImplementedError

    ms.amp.auto_mixed_precision(compute_loss, amp_level=amp_level)

    if grad_reducer is None:
        grad_reducer = ops.functional.identity

    def forward_func(x, label, sizes=None):
        x /= 255.0
        if sizes is not None:
            x = ops.interpolate(x, sizes=sizes, coordinate_transformation_mode="asymmetric", mode="bilinear")
        pred = model(x)
        if use_ota:
            loss, loss_items = compute_loss(pred, label, x)
        else:
            loss, loss_items = compute_loss(pred, label)
        return loss_scaler.scale(loss), loss_items

    grad_fn = ops.value_and_grad(forward_func, grad_position=None, weights=optimizer.parameters, has_aux=True)

    def train_step(x, label, sizes=None, optimizer_update=True):
        (loss, loss_items), grads = grad_fn(x, label, sizes)
        grads = grad_reducer(grads)
        unscaled_grads = loss_scaler.unscale(grads)
        grads_finite = all_finite(unscaled_grads)
        # _ = loss_scaler.adjust(grads_finite)

        if optimizer_update:
            if grads_finite:
                loss = ops.depend(loss, optimizer(unscaled_grads))
            else:
                print("overflow, drop the step.")

        return loss, loss_items, unscaled_grads, grads_finite

    def forward_warpper(x, label, sizes=None, optimizer_update=True):
        loss, loss_items = forward_func(x, label, sizes)
        return loss, loss_items, None, Tensor([True], ms.bool_)

    return forward_warpper


if __name__ == '__main__':
    opt = get_args_train()
    # opt.hyp = opt.hyp or ('hyp.finetune.yaml' if opt.weights else 'hyp.scratch.yaml')
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run

    ms_mode = context.GRAPH_MODE if opt.ms_mode == "graph" else context.PYNATIVE_MODE
    context.set_context(mode=ms_mode, device_target=opt.device_target)
    if opt.device_target == "Ascend":
        device_id = int(os.getenv('DEVICE_ID', 0))
        context.set_context(device_id=device_id)
    # context.set_context(pynative_synchronize=True)
    # if opt.device_target == "GPU":
    #    context.set_context(enable_graph_kernel=True)
    # Distribute Train
    rank, rank_size, parallel_mode = 0, 1, ParallelMode.STAND_ALONE
    if opt.is_distributed:
        init()
        rank, rank_size, parallel_mode = get_rank(), get_group_size(), ParallelMode.DATA_PARALLEL
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=rank_size)

    opt.total_batch_size = opt.batch_size
    opt.rank_size = rank_size
    opt.rank = rank
    if rank_size > 1:
        assert opt.batch_size % opt.rank_size == 0, '--batch-size must be multiple of device count'
        opt.batch_size = opt.total_batch_size // opt.rank_size

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Train
    if not opt.evolve:
        # profiler = Profiler()
        # check_train(hyp, opt)
        train(hyp, opt)
        # profiler.analyse()
    else:
        raise NotImplementedError("Not support evolve train;")
