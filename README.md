# 昇思MindSpore技术公开课

- ***探究前沿***：解读技术热点，解构热点模型
- ***应用实践***：理论实践相结合，手把手指导开发
- ***专家解读***：多领域专家，多元解读
- ***开源共享***：课程免费，课件代码开源
- ***大赛赋能***：ICT大赛赋能课程（大模型专题第一、二期）
- ***系列课程***：大模型专题课程开展中，其他专题课程敬请期待

## 报名方式

报名链接：https://xihe.mindspore.cn/course/foundation-model-v2/introduction 

（注：参与免费课程必须报名哦！同步添加QQ群，后续课程事宜将在群内通知！）

## 大模型专题第一期&第二期（进行中）

紧跟前沿技术，解构热点大模型（如ChatGLM2、LLAMA2等）；手把手教你大模型从开发到应用全流程

课程资料归档：[link](./Season2.step_into_llm/)

### 教研团队


![img_2.png](img_2.png)

### 课前学习

- python
- 人工智能基础、深度学习基础（重点学习自然语言处理）：[MindSpore-d2l](https://openi.pcl.ac.cn/mindspore-courses/d2l-mindspore)
- MindSpore基础使用：[MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/r2.2/index.html)
- MindFormers基础使用：[MindFormers](https://www.bilibili.com/video/BV1jh4y1m7xV/?spm_id_from=333.999.0.0)



### 课程介绍

昇思MindSpore技术公开课火热开展中，面向所有对大模型感兴趣的开发者，带领大家理论结合时间，由浅入深地逐步深入大模型技术

在已经完结的第一期课程（第1讲-第10讲）中，我们从Transformer开始，解析到ChatGPT的演进路线，手把手带领大家搭建一个简易版的“ChatGPT”

正在进行的第二期课程（第11讲-）在第一期的基础上做了全方位的升级，围绕大模型从开发到应用的全流程实践展开，讲解更前沿的大模型知识、丰富更多元的讲师阵容，期待你的加入！

| 章节序号 | 章节名称 | 课程简介                                        | 视频链接 | 课件及代码链接 |
|:----:|:----:|:--------------------------------------------|:----:|:----:|
| 第一讲 | Transformer | Multi-head self-attention原理。Masked self-attention的掩码处理方式。基于Transformer的机器翻译任务训练。 | [link](https://www.bilibili.com/video/BV16h4y1W7us/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f4290) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/1.Transformer) |
| 第二讲 | BERT | 基于Transformer Encoder的BERT模型设计：MLM和NSP任务。BERT进行下游任务微调的范式。 | [link](https://www.bilibili.com/video/BV1xs4y1M72q/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/2.BERT) |
| 第三讲 | GPT | 基于Transformer Decoder的GPT模型设计：Next token prediction。GPT下游任务微调范式。 | [link](https://www.bilibili.com/video/BV1Gh411w7HC/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/3.GPT) |
| 第四讲 | GPT2 | GPT2的核心创新点，包括Task Conditioning和Zero shot learning；模型实现细节基于GPT1的改动。 |  [link](https://www.bilibili.com/video/BV1Ja4y1u7xx/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/4.GPT2) |
| 第五讲 | MindSpore自动并行 | 以MindSpore分布式并行特性为依托的数据并行、模型并行、Pipeline并行、内存优化等技术。 |  [link](https://www.bilibili.com/video/BV1VN41117AG/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/5.Parallel) |
| 第六讲 | 代码预训练 | 代码预训练发展沿革。Code数据的预处理。CodeGeex代码预训练大模型。      |  [link](https://www.bilibili.com/video/BV1Em4y147a1/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/6.CodeGeeX) |
| 第七讲 | Prompt Tuning | Pretrain-finetune范式到Prompt tuning范式的改变。Hard prompt和Soft prompt相关技术。只需要改造描述文本的prompting。 | [link](https://www.bilibili.com/video/BV1Wg4y1K77R/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/7.Prompt) |
| 第八讲 | 多模态预训练大模型 | 紫东太初多模态大模型的设计、数据处理和优势；语音识别的理论概述、系统框架和现状及挑战。 | [link](https://www.bilibili.com/video/BV1wg4y1K72r/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | / |
| 第九讲 | Instruct Tuning | Instruction tuning的核心思想：让模型能够理解任务描述（指令）。Instruction tuning的局限性：无法支持开放域创新性任务、无法对齐LM训练目标和人类需求。Chain-of-thoughts：通过在prompt中提供示例，让模型“举一反三”。 | [link](https://www.bilibili.com/video/BV1cm4y1e7Cc/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/8.Instruction) |
| 第十讲 | RLHF | RLHF核心思想：将LLM和人类行为对齐。RLHF技术分解：LLM微调、基于人类反馈训练奖励模型、通过强化学习PPO算法实现模型微调。 | [link](https://www.bilibili.com/video/BV15a4y1c7dv/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season1.step_into_chatgpt/9.RLHF) |
| 第十一讲 | ChatGLM | 介绍技术公开课整体课程安排；ChatGLM模型结构，走读代码演示ChatGLM推理部署 | [link](https://www.bilibili.com/video/BV1ju411T74Y/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season2.step_into_llm/01.ChatGLM) |
| 第十二讲 | 多模态遥感智能解译基础模型 | 介绍多模态遥感智能解译基础模型的原理、训推等相关技术，以及模型相关行业应用       | [link](https://www.bilibili.com/video/BV1Be41197wY/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) |/|
| 第十三讲 | ChatGLM2 | 介绍ChatGLM2模型结构，走读代码演示ChatGLM推理部署            | [link](https://www.bilibili.com/video/BV1Ew411W72E/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season2.step_into_llm/02.ChatGLM2) |
| 第十四讲 | 文本生成解码原理 | 介绍Beam search和采样的原理及代码实现                    | [link](https://www.bilibili.com/video/BV1QN4y117ZK/?spm_id_from=333.999.0.0&vd_source=eb3a45e6eb4dccc5795f97586b78f429) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season2.step_into_llm/03.Decoding) |
| 第十五讲 | LLAMA | 介绍LLAMA模型结构，走读代码演示推理部署，介绍Alpaca             | [link](https://www.bilibili.com/video/BV1nN41157a9/?spm_id_from=333.999.0.0) | [link](https://github.com/mindspore-courses/step_into_llm/tree/master/Season2.step_into_llm/04.LLaMA) |
| 第十六讲 | LLAMA2 | 介绍LLAMA2模型结构，走读代码演示LLAMA2 chat部署            |
| 第十七讲 | 云从大模型 | /                                           |
| 第十八讲 | MOE | /                                           |
| 第十九讲 | CPM | 介绍CPM-Bee预训练、推理、微调及代码现场演示  |                                                                                                                                    |
| 第二十讲 | 高效参数微调 | 介绍Lora、（P-Tuning）原理及代码实现  |
| 第二十一讲 | 参数微调平台 | / |
| 第二十二讲 | Prompt Engineering | / |
| 第二十三讲 | 量化 | 介绍低比特量化等相关模型量化技术  |
| 第二十四讲 | 框架LangChain模块解析 | 解析Models、Prompts、Memory、Chains、Agents、Indexes、Callbacks模块，及案例分析 |
| 第二十五讲 | LangChain对话机器人综合案例 | MindSpore Transformers本地模型与LangChain框架组合使用，通过LangChain框架管理向量库并基于向量库对MindSpore Transformers本地模型问答进行优化 |


### 昇思资源一览：生态与伙伴共建、共享、共荣

![img_1.png](img_1.png)