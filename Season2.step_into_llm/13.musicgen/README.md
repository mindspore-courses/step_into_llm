# MusicGEN案例的运行方式分两种：

## 1、执行notebook运行

在华为云打开notebook文档，按照步骤执行即可
例如AI Gallery的在线体验案例
https://pangu.huaweicloud.com/gallery/asset-detail.html?id=c72241ed-465f-418d-b58a-ed4aabb0eb73

## 2、可交互运行：

终端运行命令如下：

conda create -n ms python=3.9 -y

conda activate ms

wget https://mindspore-demo.obs.cn-north-4.myhuaweicloud.com/mindnlp_install/mindnlp-0.3.1-py3-none-any.whl

pip install mindnlp-0.3.1-py3-none-any.whl

git clone https://github.com/mindspore-lab/mindnlp

cd mindnlp/llm/inference/musicgen
pip install gradio
python app_zh.py


**如果使用mindspore2.6.0和mindnlp0.4.0运行run_musicgen.ipynb时则需要对mindnlp源码作以下修改**
- 1. vim ~/anaconda3/envs/ms/lib/python3.9/site-packages/mindnlp/core/ops/creation.py
注释这行代码：#from mindspore._c_expression import Tensor as CTensor # pylint: disable=no-name-in-module, import-error
修改为：from mindspore import Tensor as CTensor
- 2. vim ~/anaconda3/envs/ms/lib/python3.9/site-packages/mindnlp/transformers/models/jamba/modeling_jamba.py
#from mindspore._c_expression import Tensor as RawTensor # pylint: disable=no-name-in-module
from  mindspore  import Tensor as RawTenso
- 3. vim ~/anaconda3/envs/ms/lib/python3.9/site-packages/mindnlp/core/ops/creation.py
找到empty函数注释并修改为下面这个函数：
def empty(*size, dtype=None):
    import mindspore as ms
    from mindspore import dtype as mstype
    import numpy as np

    # 处理size参数
    if isinstance(size[0], (tuple, list)):
        size = size[0]

    # 处理dtype参数
    if dtype is None:
        dtype = mstype.float32
    elif isinstance(dtype, (float, int)):
        dtype = mstype.float32
    elif not isinstance(dtype, mstype.Type):
        try:
            dtype = mstype.pytype_to_dtype(dtype)
        except:
            dtype = mstype.float32

    # 使用numpy作为中介创建空数组
    np_array = np.empty(size, dtype=np.float32)
    return ms.Tensor(np_array, dtype=dtype)
如果使用的是CPU还需要进行4.5.操作
- 4. vim ~/anaconda3/envs/ms/lib/python3.9/site-packages/mindnlp/core/ops/other.py
找到repeat_interleave函数注释并修改为如下函数：
def repeat_interleave(input, repeats, dim=None):
    if dim is None:
        input = input.flatten()
        dim = 0
    if isinstance(repeats, int):
        if input.dtype == ms.bool_:
            return mindspore.Tensor(np.tile(input.asnumpy(), (repeats,) + (1,) * (len(input.shape) - 1)), dtype=mindspore.bool_)
        else:
            return mindspore.numpy.tile(input, (repeats,) + (1,) * (len(input.shape) - 1))
    else:
        raise NotImplementedError("Only integer `repeats` is supported.")
- 5. vim ~/anaconda3/envs/ms/lib/python3.9/site-packages/mindnlp/transformers/modeling_attn_mask_utils.py
找到 _make_causal_mask函数注释并修改为下面的函数：
def _make_causal_mask(
    input_ids_shape,
    dtype,  # 注意：调用方必须保证传入正确的dtype
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    
    # 1. 创建初始mask（直接使用0/1而不是极值）
    mask = ops.zeros((tgt_len, tgt_len), dtype=dtype)
    mask_cond = ops.arange(tgt_len)
    
    # 2. 使用三角矩阵生成因果mask
    causal_mask = ops.triu(
        ops.ones((tgt_len, tgt_len), dtype=mstype.float32), 
        diagonal=1
    ).to(dtype)  # 先创建float矩阵再转换
    
    # 3. 合并past key values长度
    if past_key_values_length > 0:
        mask = ops.cat([
            ops.zeros((tgt_len, past_key_values_length), dtype=dtype),
            causal_mask
        ], dim=-1)
    else:
        mask = causal_mask

    # 4. 滑动窗口处理（保持类型一致）
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1
        context_mask = ops.triu(
            ops.ones((tgt_len, tgt_len + past_key_values_length), dtype=mstype.float32),
            diagonal=diagonal
        ).to(dtype)
        mask = mask + context_mask  # 使用加法代替masked_fill

    return mask[None, None, :, :].broadcast_to(
        (bsz, 1, tgt_len, tgt_len + past_key_values_length)
    )