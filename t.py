import mindspore as ms
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from mindyolo.models.loss import ComputeLossOTA as Loss1
from mindyolo.models.loss2 import ComputeLossOTA as Loss2
ms.set_seed(2)

# anchors = np.random.uniform(0 ,300, (3, 3, 2)).astype(np.int32)
# p = tuple([ms.Tensor(np.random.uniform(0 ,1, (32 ,3, 20 * 2**i, 20 * 2**i, 85)).astype(np.float32)) for i in range(2, -1 ,-1)])
# targets = ms.Tensor(np.random.uniform(0 ,1, (32, 700, 6)).astype(np.float32))
# imgs = ms.Tensor(np.random.uniform(0 ,1, (32 ,3, 640, 640)).astype(np.float32))

from mindyolo.models.yolo import Model
from mindyolo.utils.config import parse_args

opt = parse_args('train')
model = Model(opt, ch=3, nc=int(opt.nc), sync_bn=False)
anchors = np.load("anchors.npy")
p = tuple([ms.Tensor(np.load(f"p_{i}.npy")[:8]) for i in range(3)])
targets = ms.Tensor(np.load("targets.npy")[:8])
imgs = ms.Tensor(np.load("imgs.npy")[:8])

bt2 = Loss2(model)
y2 = bt2(p, targets, imgs)
print(y2)
bt = Loss1(model)
y1 = bt(p, targets, imgs)
print(y1)
#
# class Test:
#     def __init__(self, thread_num=4):
#         self.x = [[] for i in range(3)]
#         self.pool = ThreadPoolExecutor(thread_num)
#
#     def test(self, n):
#         for i in range(3):
#             t = np.random.randint(1, 3)
#             print(i, [n] * t)
#             self.x[i].append([n] * t)
#         return True
#
#     def __call__(self,):
#         futures = []
#         for batch_idx in range(5):
#             futures.append(self.pool.submit(self.test, batch_idx))
#         for future in futures:
#             future.result()
#         print(self.x)
#
#
# t = Test(2)
# t()