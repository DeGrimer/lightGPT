import os
import multiprocessing as mp
import numpy as np
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenize_With_Tokenizer import RusTokPreTokenizer
from tqdm import tqdm
from datasets import load_dataset
import random
tokenizer = Tokenizer.from_file("tokenizer-wiki50k.json")
tokenizer.pre_tokenizer = PreTokenizer.custom(RusTokPreTokenizer())
local_dir = "lurk"
shard_size = int(5e6)
eot = 0

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

with open('E:\\lurk\output2.txt', 'r', encoding="utf-8") as f:
            lurk = f.read()
lurk = lurk.split('<|endoftext|>')
ds = load_dataset("Spierocho/channel_posts")
posts = ds['train']['0']
lurk.extend(posts)
random.shuffle(lurk)
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(tokenizer.encode(doc).ids)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)
if __name__ == '__main__':   
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, lurk, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"lurk_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"lurk_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])