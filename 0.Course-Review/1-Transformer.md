# Transformer 课程回顾

## 注意力机制

- 注意力分数用来表示词元在序列中的重要性，分数越高，说明词元与任务的关联越强
- scaled dot-product attention计算：$\text{Attention}(Q, K) = \text{softmax}(\frac{QK^T}{\sqrt{d_{model}}})$
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

<div align="center"><img src="./assets/OOP+FP.png" alt="oop+fp"></div>

## BLEU Score

- BELU Score：衡量生成文本与参考文本之间的相似度，分数越高，翻译效果越好


