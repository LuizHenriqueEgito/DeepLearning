import pandas as pd
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from transformers import PreTrainedTokenizerFast
from tokenizers import (
    Tokenizer,
    pre_tokenizers,
    normalizers,
    trainers,
    decoders,
    models
)

from configs import VOCAB_SIZE


class SpecialTokensStr(Enum):
    PAD = '[PAD]'
    CLS = '[CLS]'
    UNK = '[UNK]'
    MASK = '[MASK]'
    SOS = '[SOS]'
    EOS = '[EOS]'

    @classmethod
    def todict(cls):
        return {f'{token.name.lower()}_token': token.value for token in cls}

    @classmethod
    def tolist(cls):
        return list(cls.todict())

class SpecialTokensInt(Enum):
    PAD = 0
    CLS = 1
    UNK = 2
    MASK = 3
    SOS = 4
    EOS = 5

    @classmethod
    def todict(cls):
        return {token.name: token.value for token in cls}

    @classmethod
    def tolist(cls):
        return list(cls.todict())

@dataclass
class TokenizerImDB:
    vocab_size: int
    tokenizer_path: Path
    tokenizer: Tokenizer = None

    def __post_init__(self):
        if self.tokenizer_path.exists():
            self.load_tokenizer()

    def train(self, text_iterator):
        self.tokenizer = Tokenizer(models.BPE())

        self.tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Lowercase(),
                normalizers.Strip(),
                normalizers.NFC(),
                normalizers.NFD(),
                normalizers.NFKC(),
                normalizers.NFKD(),
            ]
        )

        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        self.tokenizer.decoder = decoders.ByteLevel()

        special_tokens = SpecialTokensStr.tolist()

        self.trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size, 
            special_tokens=special_tokens
        )

        self.tokenizer.train_from_iterator(text_iterator, self.trainer)

        self.tokenizer.save(str(self.tokenizer_path))

    def encoder(self, text: str, **kwargs) -> list[int]:
        return self.tokenizer.encode(text, **kwargs).ids

    def decoder(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def load_tokenizer(self):
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))


def text_iterator(dataset: pd.DataFrame, language: str):
    match language:
        case 'pt':
            text_col = dataset['text_pt']
        case 'en':
            text_col = dataset['text_en']
    for text in text_col:
        yield text


if __name__ == '__main__':
    file_dataset = Path('../data/imdb-reviews-pt-br.csv')
    dataset = pd.read_csv(file_dataset)
    tokenizer_path_pt = Path('artifacts/tokenizer_pt.json')
    tokenizer_path_en = Path('artifacts/tokenizer_en.json')

    tokenizer_pt = TokenizerImDB(vocab_size=VOCAB_SIZE, tokenizer_path=tokenizer_path_pt)
    tokenizer_en = TokenizerImDB(vocab_size=VOCAB_SIZE, tokenizer_path=tokenizer_path_en)