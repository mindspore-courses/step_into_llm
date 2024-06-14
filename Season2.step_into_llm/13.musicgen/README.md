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