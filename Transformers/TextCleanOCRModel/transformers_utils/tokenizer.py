"""O encoder usado aqui deve ser a nível de caractere, pois queremos remover sequências de caracteres que não façam sentido."""

import yaml
import string
import unidecode
from typing import List

with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)

class Tokenizer:
    def __init__(self):
        self.vocab_size = len(string.printable)
        self.vocab  =string.printable

    def normalize_text(self, text):
        return unidecode.unidecode(text)

    def encode_text_to_id(self, text: str) -> List[int]:
        text = self.normalize_text(text)
        ids_list = [string.printable.index(char) for char in text] + [settings['EOS_TOKEN']]
        diff_to_max_seq =  settings['MAX_SEQ_LEN'] - len(ids_list)
        if diff_to_max_seq >= 0:
            return ids_list + [settings['PAD_TOKEN']] * diff_to_max_seq
        else:
            return ids_list[:settings['MAX_SEQ_LEN'] - 1] + [settings['EOS_TOKEN']]

    def decode_id_to_text(self, list_ids: List[int]) -> str:
        return  ''.join([string.printable[id_] for id_ in list_ids])
