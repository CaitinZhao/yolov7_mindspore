import time
import numpy as np
import mindspore as ms
from mindspore import ops


def box_iou_np(box1, box2):
    """
    Calculate the iou of box1 and box2 with numpy.
    Args:
        box1 (ndarray): [N, 4]
        box2 (ndarray): [M, 4], usually N != M
    Return:
        iou (ndarray): iou of box1 and box2, [N, M]
    """
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(
        box1[:, None, :2], box2[:, :2])).clip(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def topk(x, k):
    x_shape = x.shape
    out_shape = x_shape[:-1] + (k,)
    matrix = x.reshape((-1, x_shape[-1]))
    c, n = matrix.shape
    index_part = np.argpartition(matrix, -k)[:, -k:]
    index_channel = np.arange(c)[:, None]
    part_sort_k = np.argsort(matrix[index_channel, index_part], axis=-1)
    top_k_index = np.flip(index_part[index_channel, part_sort_k], axis=-1)
    top_k_scores = matrix[index_channel, top_k_index].reshape(out_shape)
    return top_k_scores, top_k_index.reshape(out_shape)


def sigmoid(x):
    y = 1 / (1 + (np.exp((-x))))
    return y


def one_hot(x, num_clases):
    return np.eye(num_clases)[x.astype(np.int32)]


def binary_cross_entropy_with_logits(x, y):
    x = sigmoid(x)
    return -(y * np.log(x) + (1 - y) * np.log(1 - x))


class BuildTarget:
    """
    Build target of YOLOv7
    Args:
        anchors (Tensor): anchors in config
        na (int): channel numbers
        bias (float): bias in find positive
        stride (list): stride list of YOLO out's feature
        anchor_t (float): anchor_t thr
    Inputs:
        p (list(Tensor)): predicts(layer_num, batch_size, anchors_num, feature_size_h, feature_size_w, class_num+1+4).
                    1 is positive object predict, 4 is x, y, w, h
        targets (Tensor): targets(batch_size, pre_batch_target_num, 6). 6 is image_index, cls_id, x, y, w, h
        img (Tensor): input image
    """

    def __init__(self, anchors, na=3, bias=0.5, stride=[8, 16, 32], anchor_t=4, use_aux=False, use_static=False):
        super(BuildTarget, self).__init__()
        if isinstance(anchors, ms.Tensor):
            anchors = anchors.asnumpy()
        if isinstance(stride, ms.Tensor):
            stride = stride.asnumpy()
        self.anchors = anchors
        self.na = na
        self.bias = bias
        self.stride = stride
        self.anchor_t = anchor_t
        self.off = np.array(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
            ],
            dtype=np.float32) * bias  # offsets
        self.use_aux = use_aux
        self.use_static = use_static

    def find_positive(self, outputs, targets, all_anchors, g=0.5):
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        indices, anch = [], []
        gain = np.ones(7, dtype=np.float32)  # normalized to gridspace gain
        ai = np.tile(np.arange(na, dtype=np.float32).reshape(na, 1), [1, nt])
        targets = np.concatenate((np.tile(
            np.expand_dims(targets, 0), [na, 1, 1]), ai[:, :, None]), 2)

        for i in range(len(all_anchors)):
            anchors = all_anchors[i]
            gain[2:6] = np.array(outputs[i].shape, dtype=np.float32)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets_labels to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = np.maximum(r, 1. / r).max(2) < self.anchor_t
                if not np.any(j):
                    t = targets[0]
                    offsets = 0
                    continue
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = np.stack([np.ones_like(j), j, k, l, m])
                t = np.tile(t, [5, 1, 1])[j]
                offsets = (np.zeros_like(gxy)[None] + self.off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].astype(np.int64).T
            gxy = t[:, 2:4]  # grid xy
            gij = (gxy - offsets).astype(np.int64)
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].astype(np.int64)  # anchor indices
            gj, gi = gj.clip(0, gain[3] - 1).astype(np.int64), gi.clip(
                0, gain[2] - 1).astype(np.int64)
            indices.append((b, a, gj, gi))
            anch.append(anchors[a])  # anchors
        # return numpy rather than tensor
        return indices, anch

    def build_targets(self, p, targets, imgs, min_topk=10, g=0.5):
        s = time.time()
        targets = targets.asnumpy().astype(np.float32)
        p = [pp.asnumpy() for pp in p]
        print("asnumpy", time.time() - s)

        targets = targets.reshape((-1, 6))
        targets = targets[targets[:, 1] > 0]
        indices, anch = self.find_positive(p, targets, self.anchors, g)
        # numpy indices,anch for fast assign

        matching_bs = [[] for _ in p]
        matching_as = [[] for _ in p]
        matching_gjs = [[] for _ in p]
        matching_gis = [[] for _ in p]
        matching_targets = [[] for _ in p]
        matching_anchs = [[] for _ in p]

        nl = len(p)
        for batch_idx in range(p[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            if b_idx.sum() == 0:
                continue
            this_target = targets[b_idx]
            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            # this_target[:, 2:6] * 640
            txyxy = xywh2xyxy(txywh)  # tensor op

            pxyxys, p_cls, p_obj = [], [], []
            from_which_layer = []
            all_b, all_a, all_gj, all_gi = [], [], [], []
            all_anch = []

            empty_feats_num = 0

            for i, pi in enumerate(p):
                b, a, gj, gi = indices[i]
                idx = (b == batch_idx)
                if idx.sum() == 0:
                    empty_feats_num += 1
                    continue
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_anch.append(anch[i][idx])
                from_which_layer.append(np.ones([len(b)]) * i)

                fg_pred = pi[b, a, gj, gi]  # numpy index
                if len(fg_pred.shape) == 1:  # Note: when only one sample
                    fg_pred = fg_pred[None, :]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = np.stack([gi, gj], 1)
                pxy = (sigmoid(fg_pred[:, :2]) * 2. - 0.5 + grid) * self.stride[i]
                anch_inx = anch[i][idx]
                pwh = (sigmoid(fg_pred[:, 2:4]) * 2) ** 2 * anch_inx * self.stride[i]
                pxywh = np.concatenate([pxy, pwh], -1)
                pxyxy = xywh2xyxy(pxywh)
                pxyxys.append(pxyxy)

            if empty_feats_num == len(p) or len(pxyxys) == 0:  # Note: empty
                continue
            pxyxys = np.concatenate(pxyxys, 0)

            p_obj = np.concatenate(p_obj, 0)
            p_cls = np.concatenate(p_cls, 0)

            from_which_layer = np.concatenate(from_which_layer, 0)
            all_b = np.concatenate(all_b, 0)
            all_a = np.concatenate(all_a, 0)
            all_gj = np.concatenate(all_gj, 0)
            all_gi = np.concatenate(all_gi, 0)
            all_anch = np.concatenate(all_anch, 0)

            pairwise_ious = box_iou_np(txyxy, pxyxys)
            # [N, 4] [M, 4] to get [N, M] ious

            pairwise_iou_loss = -np.log(pairwise_ious + 1e-8)

            topk_ious, _ = topk(pairwise_ious, min(min_topk, pairwise_ious.shape[1]))
            dynamic_ks = np.maximum(topk_ious.sum(1).astype(np.int32), 1)
            # this_target: (6,) image_index, cls_id, x, y, w, h
            # gt_cls_per_image: (target_num, M, class_num)
            gt_cls_per_image = np.tile(one_hot(this_target[:, 1], p_cls.shape[-1])[:, None, :],
                                       [1, pxyxys.shape[0], 1])

            num_gt = this_target.shape[0]
            cls_preds = (
                    sigmoid(np.tile(p_cls[None, :, :], [num_gt, 1, 1])) *
                    sigmoid(np.tile(p_obj[None, :, :], [num_gt, 1, 1])))

            y = np.sqrt(cls_preds + 1e-8)
            pairwise_cls_loss = binary_cross_entropy_with_logits(
                np.log(y / (1 - y)), gt_cls_per_image).sum(-1)

            cost = (pairwise_cls_loss + 3.0 * pairwise_iou_loss)

            matching_matrix = np.zeros(cost.shape)
            for gt_idx in range(num_gt):
                _, pos_idx = topk(cost[gt_idx], k=dynamic_ks[gt_idx])
                matching_matrix[gt_idx, pos_idx] = 1.0

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum() > 0:
                cost_argmin = np.argmin(cost[:, anchor_matching_gt > 1], 0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_anch = all_anch[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_targets[i].append(this_target[layer_idx])
                matching_anchs[i].append(all_anch[layer_idx])
        if not self.use_static:
            t = time.time()
            for i in range(nl):
                if matching_targets[i] != []:
                    matching_bs[i] = ms.Tensor.from_numpy(np.concatenate(matching_bs[i], 0))
                    matching_as[i] = ms.Tensor.from_numpy(np.concatenate(matching_as[i], 0))
                    matching_gjs[i] = ms.Tensor.from_numpy(np.concatenate(matching_gjs[i], 0))
                    matching_gis[i] = ms.Tensor.from_numpy(np.concatenate(matching_gis[i], 0))
                    matching_targets[i] = ms.Tensor.from_numpy(np.concatenate(matching_targets[i], 0))
                    matching_anchs[i] = ms.Tensor.from_numpy(np.concatenate(matching_anchs[i], 0))
                else:
                    matching_bs[i] = ms.Tensor.from_numpy(np.array([]))
                    matching_as[i] = ms.Tensor.from_numpy(np.array([]))
                    matching_gjs[i] = ms.Tensor.from_numpy(np.array([]))
                    matching_gis[i] = ms.Tensor.from_numpy(np.array([]))
                    matching_targets[i] = ms.Tensor.from_numpy(np.array([]))
                    matching_anchs[i] = ms.Tensor.from_numpy(np.array([]))
            print("to tensor", time.time() - t)
            print("build target", time.time() - s)
            return matching_bs, matching_as, matching_gjs, matching_gis, matching_targets, matching_anchs
        classes, objs, boxes = [], [], []

        for i in range(nl):
            b, a, h, w, n = p[i]
            cls, obj, box = np.zeros((b, a, h, w, n), np.float32),\
                            np.zeros((b, a, h, w, n), np.float32), np.zeros((b, a, h, w, n), np.float32)
            classes.append(cls)
            objs.append(obj)
            boxes.append(box)
        return classes, objs, boxes

    def __call__(self, p, targets, imgs):
        if not self.use_static:
            if not self.use_aux:
                bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs, min_topk=10, g=self.bias)
                return bs, as_, gjs, gis, targets, anchors
            bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux = self.build_targets(
                p, targets, imgs, min_topk=20, g=self.bias*2)
            bs, as_, gjs, gis, targets, anchors = self.build_targets(p, targets, imgs, min_topk=10, g=self.bias)
            return bs_aux, as_aux_, gjs_aux, gis_aux, targets_aux, anchors_aux, bs, as_, gjs, gis, targets, anchors
        else:
            if not self.use_aux:
                classes, objs, boxes = self.build_targets(p, targets, imgs, min_topk=10, g=self.bias)
                return classes, objs, boxes
            classes_aux, objs_aux, boxes_aux = self.build_targets(
                p, targets, imgs, min_topk=20, g=self.bias * 2)
            classes, objs, boxes = self.build_targets(p, targets, imgs, min_topk=10, g=self.bias)
            return classes_aux, objs_aux, boxes_aux, classes, objs, boxes
