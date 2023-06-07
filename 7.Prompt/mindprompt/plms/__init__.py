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
from typing import List, Optional
from collections import namedtuple

from mindnlp.abc.models import PreTrainedModel
from mindnlp.abc.transforms import PreTrainedTokenizer
from mindnlp.models import GPT2Config, GPT2LMHeadModel
from mindnlp.transforms.tokenizers import GPT2Tokenizer
from mindspore import log as logger

from .lm import LMTokenizerWrapper

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model', 'wrapper'))

_MODEL_CLASSES = {
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
}


def get_model_class(plm_type: str):
    return _MODEL_CLASSES[plm_type]


def load_plm(model_name, model_path, specials_to_add=None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_class = get_model_class(plm_type=model_name)
    model_config = model_class.config.from_pretrained(model_path)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name:  # add pad token for gpt
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained(model_path, config=model_config, from_pt=False)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper

    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    if 'opt' in model_name:
        tokenizer.add_bos_token = False
    return model, tokenizer, model_config, wrapper


def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.

    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer
