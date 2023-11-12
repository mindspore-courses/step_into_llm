import os
import platform
import signal
import readline

import time
import mindspore as ms
from mindformers import AutoConfig, AutoModel, AutoTokenizer


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=0)

config = AutoConfig.from_pretrained("glm2_6b")
# 可以在此使用下行代码指定自定义权重进行推理，默认使用自动从obs上下载的预训练权重
# config.checkpoint_name_or_path = "/path/to/glm2_6b_finetune.ckpt"
config.use_past = True
model = AutoModel.from_config(config)
tokenizer = AutoTokenizer.from_pretrained("glm2_6b")

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM2-6B：{response}"
    return prompt


def signal_handler(signal, frame):
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM2-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0
        
        # prompt: [Round 1]\n\n问：{query}\n\n答： 
        prompted_inputs = tokenizer.build_prompt(query)
        inputs = tokenizer(prompted_inputs)['input_ids']
        outputs = model.generate(inputs, max_length=128)
        response = tokenizer.decode(outputs)[0].split("答： ")[-1]
        history = history + [(query, response)]

        if stop_stream:
            stop_stream = False
            break
        else:
            count += 1
            if count % 8 == 0:
                os.system(clear_command)
                print(build_prompt(history), flush=True)
                signal.signal(signal.SIGINT, signal_handler)
        os.system(clear_command)
        print(build_prompt(history), flush=True)


if __name__ == "__main__":
    main()