# Transformer 课程回顾

提到大型语言模型（LLM），我们不可避免地提到Transformer模型。以它为起点，自然语言处理（NLP）领域步入了新的篇章。

Transformer使用纯自注意力机制替代了传统的RNN与CNN网络，极大地提升了模型在处理长文本和长距离依赖关系的效果和速度，为后续大型语言模型的发展奠定了基础。

接下来我们对课程进行简单回顾，迎接下一节公开课的进一步深入。

## 注意力机制

- 注意力分数用来表示词元在序列中的重要性，分数越高，说明词元与任务的关联越强
- scaled dot-product attention计算：$$\text{Attention}(Q, K) = \text{softmax}(\frac{QK^T}{\sqrt{d_{model}}})$$
- 自注意力分数表示一个序列中，词元与词元之间的关系，query=key=value
- 多头注意力从多方面捕捉输入内容特征，支持并行计算注意力分数

## Transformer

- Encoder-Decoder结构
- Encoder负责抓取源序列的特征信息，并传递给Decoder，Decoder逐词输出翻译结果
- 序列在输入前需通过位置编码添加位置信息，此处的编码信息是固定的，不会随模型优化而更新
- EncoderLayer由多头注意力和前馈神经网络两个子层组成，中间进行残差连接与层归一化
- DecoderLayer由两个多头注意力与一个前馈神经网络，共三个子层组成，中间进行残差连接与层归一化
- DecoderLayer的多头子注意力需要额外添加掩码，表示它无法看到后面的词元

## NLP中的数据预处理

- 词典：收录输入中包含的词元，并将其映射为数字编码

## MindSpore OOP+FP 混合编程

<div align="center"><img src="./assets/1.transformer/OOP+FP.png" alt="oop+fp"></div>

## BLEU Score

- BELU Score：衡量生成文本与参考文本之间的相似度，分数越高，翻译效果越好

# 课程实践

- 尝试使用混合精度，提升模型训练及推理速度（包括BLEU Score计算）；
- 尝试更换数据集进行另两种语言的机器翻译；

