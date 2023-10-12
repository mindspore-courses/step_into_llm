# BERT 课程回顾

如果说Transformer是照亮了语言模型全兴发展道路的一盏等，那么BERT便是这条路上的第一个路标。

BERT在模型方面做出的改动并不复杂，但由此引出的观点却是具有颠覆性的。从BERT这一阶段的模型开始，我们将计算机视觉（CV）领域的预训练-微调的方式引入自然语言处理（NLP）中，从而摒弃的task-specific模式，使得模型适应下游任务应用的成本大幅降低。

接下来我们对课程进行简单回顾，迎接下一节公开课的进一步深入。

## 语言模型衍变

- Language Model占比逐渐变大，自然语言处理研究逐渐由task-specific转变为general purpose

## BERT介绍

- Encoder-only结构
- 预训练过程中，通过填充词语任务（Masked LM）与下一句预测任务（NSP）学习文本的双向表示
- 下游任务中，将BERT Encoder部分直接用于输入编码，对输出层进行微调
- BERT输入表示：token embeddings + segment embeddings + position embeddings
- 掩蔽语言模型（Masked LM）：捕捉词语级别的信息，模型根据上下文预测被遮盖（masked）的输入单词
- 掩蔽语言模型（Masked LM）任务中的小技巧：训练中，只需要计算被遮盖单词的预测结果，无需计算全部词元的预测输出
- 下一句预测（NSP）：二分类任务，捕捉句子级别的信息，通过预测<cls>的输出，判断两个句子是否相连

## BERT 预训练

- 数据并行：(单机多卡场景)将大批量数据切分为多个小批量，在多个设备上同时独立进行正反向传播获得梯度，再将各设备之间的梯度进行聚合，最后以相同的梯度值进行模型参数更新
- MindSpore实现数据并行：反向传播获取梯度后，调用mindspore.nn.DistributedGradReducer
- 梯度裁剪：通过限制梯度的大小来防止梯度爆炸或消失

## BERT Finetune

- BERT的下游任务：单句子分类、句子对分类、问答、文本标注
- 情感分类任务：多分类问题，在BERT预训练模型基础上添加线性层作为输出层，预测<cls>的分类结果

## 混合精度

- 在训练过程中，使用低精度数值（如float32）表示梯度和权重参数，从而提高计算效率、节约内存空间
- 梯度缩放：为避免float16无法表示梯度的微小变化，在反向传播前放大损失值，并在获得梯度值后同比例缩小
- 混合精度全流程
<div align="center"><img src="./assets/2.bert/mix_precision.png" alt="mix-precision"></div>

# 课程实践

- 参考课程中介绍的下游任务，自己选择一种进行实践；