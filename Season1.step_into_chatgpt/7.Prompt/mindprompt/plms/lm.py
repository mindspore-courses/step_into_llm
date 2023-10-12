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
from typing import Optional
from collections import defaultdict

from mindprompt.plms.utils import TokenizerWrapper
from mindnlp.abc.transforms import PreTrainedTokenizer


class LMTokenizerWrapper(TokenizerWrapper):
    r"""
    LMTokenizer is a causual language model. Therefore it can only predict <mask> position
    at the end of the sentence. A prefix-style template like: 'A <mask> news : <text_a> <text_b> ' is
    not applicable in this situation.
    For the template where there is '<text_a>' or '<text_b>' after '<mask>', we raise an exception and terminate
    the program.
    For the template where there are template words after '<mask>', we ignore these template words.
    Moreover, it can only predict one '<mask>' position. All template that has multiple '<mask>' will
    give rise to an exception.
    """

    def __init__(self,
                 max_seq_length: int,
                 tokenizer: PreTrainedTokenizer,
                 truncate_method: Optional[str] = 'tail',
                 predict_eos_token: Optional[bool] = False,
                 **kwargs):
        super().__init__(max_seq_length=max_seq_length, tokenizer=tokenizer, truncate_method=truncate_method)
        self.predict_eos = predict_eos_token

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doesn't consider the situation that input has two parts
        '''
        wrapped_example, others = wrapped_example

        if teacher_forcing:

            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]

        if self.predict_eos:
            if not wrapped_example[-1]['text'].endswith(self.tokenizer.eos_token):
                wrapped_example.append({"text": self.tokenizer.eos_token, "shortenable_ids": 0, "loss_ids": 1})

        encoder_inputs = defaultdict(list)

        num_mask_token_used = 0

        for piece_id, piece in enumerate(wrapped_example):
            if len(piece['text']) == 0:
                continue

            if piece['text'] == self.tokenizer.eos_token and self.predict_eos and wrapped_example[piece_id - 1][
                'loss_ids'] == 1:  # eos after the mask also need to be pred
                piece['loss_ids'] = 1

            if piece['text'] == self.template_mask_token:
                if teacher_forcing:
                    piece['text'] = " " + tgt_text[num_mask_token_used] + " "
                else:
                    encoder_inputs['loss_ids'][-1][-1] = 1
                    break

            if piece['text'] in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[piece['text']]
                if to_replace is not None:
                    piece['text'] = to_replace
                else:
                    raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

            if 'soft_token_ids' in piece and piece['soft_token_ids'] != 0:
                encode_text = [0]  # can be replace by any token, since these token will use their own embeddings
            else:
                # 修改
                encode_text = self.tokenizer.encode(piece['text'])

            encoding_length = len(encode_text)
            # 修改
            encoder_inputs['input_ids'].append(encode_text.ids)
            for key in piece:
                if key not in ['text']:
                    encoder_inputs[key].append([piece[key]] * encoding_length)

        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(
            encoder_inputs=encoder_inputs)  # this will do nothing in GPT2 tokenizer
        # create special input ids
        encoder_inputs['attention_mask'] = [1] * len(encoder_inputs['input_ids'])
        if self.create_token_type_ids:
            encoder_inputs['token_type_ids'] = [0] * len(encoder_inputs['input_ids'])
        # pad to max length
        input_ids_len = len(encoder_inputs['input_ids'])
        encoder_inputs = self.padding(
            input_dict=encoder_inputs,
            max_len=self.max_seq_length,
            pad_id_for_inputs=self.tokenizer.pad_token_id
        )
        encoder_inputs = {**encoder_inputs, "input_ids_len": input_ids_len}
        return encoder_inputs
