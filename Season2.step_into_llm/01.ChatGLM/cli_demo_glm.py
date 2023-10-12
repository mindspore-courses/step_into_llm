import os
import platform
import signal
import numpy as np

import mindspore as ms
from mindformers.models.glm import GLMConfig, GLMChatModel
from mindformers.models.glm.chatglm_6b_tokenizer import ChatGLMTokenizer
from mindformers.models.glm.glm_processor import process_response

config = GLMConfig(
    position_encoding_2d=True,
    use_past=True,
    is_sample_acceleration=True)

ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=0)
model = GLMChatModel(config)
# https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm/glm_6b.ckpt
ms.load_checkpoint("./glm_6b.ckpt", model)
# https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/glm/ice_text.model
tokenizer = ChatGLMTokenizer('./ice_text.model')


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM-6B：{response}"
    return prompt


def signal_handler():
    global stop_stream
    stop_stream = True


def main():
    history = []
    global stop_stream
    print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            history = []
            os.system(clear_command)
            print("欢迎使用 ChatGLM-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        count = 0

        inputs = tokenizer(query)
        outputs = model.generate(np.expand_dims(np.array(inputs['input_ids']).astype(np.int32), 0),
                                 max_length=config.max_decode_length, do_sample=False, top_p=0.7, top_k=1)

        response = tokenizer.decode(outputs)
        response = process_response(response[0])
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
