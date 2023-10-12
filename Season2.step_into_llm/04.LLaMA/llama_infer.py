# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""llama infer"""

import os
import logging
import argparse
import numpy as np
import mindspore as ms
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
from mindformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from mindformers import init_context, ContextConfig, ParallelContextConfig
from mindformers.trainer.utils import get_last_checkpoint
from mindformers.tools.utils import str2bool
from mindformers.tools.logger import logger

os.environ['GLOG_v'] = '3'
logger.setLevel(logging.ERROR)


def context_init(use_parallel=False, device_id=0):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=device_id)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                full_batch=True)
    init_context(use_parallel=use_parallel, context_config=context_config, parallel_config=parallel_config)


def main(model_type='llama_7b',
         use_parallel=False,
         device_id=0,
         checkpoint_path="",
         use_past=True):
    """main function."""
    context_init(use_parallel, device_id)

    # set model config
    model_config = AutoConfig.from_pretrained(model_type)
    model_config.use_past = use_past
    if checkpoint_path and not use_parallel:
        model_config.checkpoint_name_or_path = checkpoint_path

    # build tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    # build model from config
    network = AutoModel.from_config(model_config)

    # if use parallel, load distributed checkpoints
    if use_parallel:
        # find the sharded ckpt path for this rank
        ckpt_path = os.path.join(checkpoint_path, "rank_{}".format(os.getenv("RANK_ID", "0")))
        ckpt_path = get_last_checkpoint(ckpt_path)
        print("ckpt path: %s", str(ckpt_path))

        # shard pangualpha and load sharded ckpt
        model = Model(network)
        model.infer_predict_layout(ms.Tensor(np.ones(shape=(1, model_config.seq_length)), ms.int32))
        checkpoint_dict = load_checkpoint(ckpt_path)
        not_load_network_params = load_param_into_net(model, checkpoint_dict)
        print("Network parameters are not loaded: %s", str(not_load_network_params))

    text_generation_pipeline = pipeline(task="text_generation", model=network, tokenizer=tokenizer)

    in_txt = input('you:')
    while in_txt != 'exit':
        outputs = text_generation_pipeline(in_txt)
        for output in outputs:
            print('llama:', output['text_generation_text'][0].split('\n')[1])
        in_txt = input('you:')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='llama_7b', type=str, help='which model to use.')
    parser.add_argument('--use_parallel', default=False, type=str2bool, help='whether use parallel.')
    parser.add_argument('--device_id', default=0, type=int, help='set device id.')
    parser.add_argument('--checkpoint_path', default='', type=str, help='set checkpoint path.')
    parser.add_argument('--use_past', default=True, type=str2bool, help='whether use past.')
    args = parser.parse_args()

    main(args.model_type, args.use_parallel, args.device_id, args.checkpoint_path, args.use_past)
