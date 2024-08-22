import spacy
import torch
from enum import Enum
from typing import List, Iterable
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator


class MyTokenizer:
    def __init__(self):
        self.PAD_IDX, self.UNK_IDX, self.BOS_IDX, self.EOS_IDX, self.MASK_IDX = 0, 1, 2, 3, 4
        self.special_symbols = ['<pad>', '<unk>', '<sos>', '<eos>', '<mask>']
        self.language = 'pt'


    def create_tokenizer(self, text_iterator: Iterable[str]):
        self.tokenizer = get_tokenizer('spacy', language=f'{self.language}_core_news_sm')
        self.vocab_transform = build_vocab_from_iterator(
            self.yield_tokens(text_iterator, self.tokenizer),
            min_freq=3,
            specials=self.special_symbols,
            special_first=True,
        )
        self.vocab_transform.set_default_index(self.UNK_IDX)


    def yield_tokens(self, data_iter: Iterable[str], tokenizer) -> List[str]:
        for text in data_iter['train']['review']:
            yield tokenizer(text)

    def tokenize_text(self, text: str) -> List[int]:
        return self.sequential_transforms(
            self.tokenizer,
            self.vocab_transform,
            self.tensor_transform
        )(text)

    def untokenize_tokens(self, tokens: List[int]) -> str:
        return " ".join(self.vocab_transform.lookup_tokens(tokens)).replace("<sos>", "").replace("<eos>", "")

    def tensor_transform(self, token_ids: List[int]):
        return torch.tensor([self.BOS_IDX] + token_ids + [self.EOS_IDX])

    def sequential_transforms(self, *steps):
        def func(txt_input):
            for step in steps:
                txt_input = step(txt_input)
            return txt_input
        return func


    
