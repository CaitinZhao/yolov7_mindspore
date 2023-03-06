import torch
import mindspore as ms
import numpy as np
from torch_yolo.yolov7.utils.loss import ComputeLossOTA as Loss1
from mindyolo.models.loss2 import ComputeLossOTA as Loss2

from mindyolo.models.yolo import Model
from mindyolo.utils.config import parse_args

opt = parse_args('train')
model = Model(opt, ch=3, nc=int(opt.nc), sync_bn=False)
anchors = np.load("anchors.npy")
p = tuple([np.load(f"p_{i}.npy") for i in range(3)])
targets = np.load("targets.npy")
imgs = np.load("imgs.npy")

bt = Loss1()
p_t = [torch.from_numpy(np.load(f"p_{i}.npy")) for i in range(3)]
y1 = bt.build_targets(p_t, torch.from_numpy(targets), torch.from_numpy(imgs))

bt2 = Loss2(model)
p_m = [ms.Tensor(np.load(f"p_{i}.npy")) for i in range(3)]
y2 = bt2(p_m, ms.Tensor(targets), ms.Tensor(imgs))
