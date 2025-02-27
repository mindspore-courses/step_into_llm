import os  # 用于获取环境变量
import numpy as np
import mindspore as ms
from mindspore.nn import Cell, Momentum
from mindspore.ops import operations as ops
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore import train
import mindspore.dataset as ds
import mindspore.communication as D
from mindspore.common.initializer import initializer

# 获取设备数量的环境变量，如果没有设置，默认为1
devices = int(os.getenv('DEVICE_NUM', 2))  # 从环境变量 DEVICE_NUM 获取设备数

# 验证设备数的有效性
if devices < 1 or devices > 8:
    print('Invalid number of devices. Exiting...')
    exit(0)

step_per_epoch = 4

def get_dataset(*inputs):
    def generate():
        for _ in range(step_per_epoch):
            yield inputs
    return generate

class Net(Cell):
    """定义网络"""
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul().shard(((1, 2), (2, 1)))  # 分配计算
        self.weight = ms.Parameter(initializer("normal", [32, 16]), "w1")  # 权重初始化
        self.relu = ops.ReLU().shard(((2, 1),))  # ReLU 激活函数

    def construct(self, x):
        out = self.matmul(x, self.weight)  # 矩阵乘法
        out = self.relu(out)  # 激活函数
        return out

if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")  # 设置设备为 Ascend NPU
    D.init()
    rank = D.get_rank()  # 获取当前进程的 rank
    ms.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=devices, full_batch=True)  # 设置并行上下文

    # 数据集生成
    np.random.seed(1)
    input_data = np.random.rand(16, 32).astype(np.float32)
    label_data = np.random.rand(16, 16).astype(np.float32)
    fake_dataset = get_dataset(input_data, label_data)

    net = Net()

    # 回调函数
    callback = [train.LossMonitor(), train.ModelCheckpoint(directory="{}".format(rank))]
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])
    loss = SoftmaxCrossEntropyWithLogits()

    # 优化器
    learning_rate = 0.001
    momentum = 0.1
    epoch_size = 5
    opt = Momentum(net.trainable_params(), learning_rate, momentum)

    model = ms.Model(net, loss_fn=loss, optimizer=opt)
    model.train(epoch_size, dataset, callbacks=callback, dataset_sink_mode=False)
