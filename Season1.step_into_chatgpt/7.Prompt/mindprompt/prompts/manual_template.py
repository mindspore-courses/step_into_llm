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
from typing import *

from mindnlp.abc.transforms import PreTrainedTokenizer

from mindprompt import Template


class ManualTemplate(Template):
    """
    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
        placeholder_mapping (:obj:`dict`): A place holder to represent the original input text. Default to ``{'<text_a>': 'text_a', '<text_b>': 'text_b'}``
    """

    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>': 'text_a', '<text_b>': 'text_b'},
                 ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.text = text

    def on_text_set(self):
        """
        when template text was set

        1. parse text
        """

        self.text = self.parse_text(self.text)
