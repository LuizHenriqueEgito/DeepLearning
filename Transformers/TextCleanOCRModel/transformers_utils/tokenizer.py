import yaml
from typing import List
from transformers import AutoTokenizer


# tokenizer pt-br utilizado ("lucas-leme/FineBERT-PT-BR")
tokenizer_path = "lucas-leme/FinBERT-PT-BR"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

with open('settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)

class Tokenizer:
    def __init__(self, max_seq_len: int = settings['MAX_SEQ_LEN'], tokenizer = tokenizer, use_pad: bool = False):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.use_pad = use_pad

    def tokenize_text_to_id(self, text: str) -> List[int]:
        """
        Função acrescenta o PAD e tokeniza o texto
        """
        text_tokenize = self.tokenizer.tokenize(text)
        diff_len = self.max_seq_len - len(text_tokenize)
        if self.use_pad:
            if diff_len > 0:
                text_tokenize += ["[PAD]"] * diff_len
            else:
                text_tokenize = text_tokenize[:self.max_seq_len]
        return self.tokenizer.convert_tokens_to_ids(text_tokenize)

    def tokenize_id_to_text(self, ids_list: List[int]) -> str:
        """
        Função que transforma a lista de ids em texto
        """
        token_list = self.tokenizer.convert_ids_to_tokens(ids_list)
        text = self.tokenizer.convert_tokens_to_string(token_list)
        return text
        
        