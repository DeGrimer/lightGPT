from tokenizers import Tokenizer, NormalizedString, PreTokenizedString
from tokenizers.pre_tokenizers import PreTokenizer, Whitespace
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer
from typing import List
import regex as re


files = [f"input.txt"] # Articles from lurk

special_tokens=["<|endoftext|>"]


class RusTokPreTokenizer:
    def rus_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        pattern = r""" ?(?i:кое|кой)-|-(?:либо|нибудь|то|таки)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        text_chunkes = re.finditer(pattern, str(normalized_string)) # -> MatchObject
        chunks_index = [(chunk.start(), chunk.end()) for chunk in text_chunkes]
        for ch in chunks_index:
            splits.append(normalized_string[ch[0]:ch[1]])
        return splits
    def pre_tokenize(self, pretok: PreTokenizedString):
        # Here we need to call 'pretok.split' with splitting methos as parameters
        pretok.split(self.rus_split)

if __name__ == '__main__':
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = PreTokenizer.custom(RusTokPreTokenizer())
    tokenizer.decoder = BPEDecoder()
    trainer = BpeTrainer(vocab_size=50257,special_tokens=special_tokens)
    tokenizer.train(files,trainer)

    tokenizer.pre_tokenizer = Whitespace() 
    # Since you can't save with a custom pre_tokenizer, you need to replace it with a standard one before saving.
    tokenizer.save("tokenizer-wiki50k.json")
    print(f"Trained vocab size: {tokenizer.get_vocab_size()}")