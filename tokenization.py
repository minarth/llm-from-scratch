## Chapter 2
from pprint import pprint

with open("data/the-verdict.txt", "r") as fd:
    raw_text = fd.read()

import re

# r = re.split(r"(\s)", raw_text)   # this splits into words like "that, "
# r = raw_text.split()

# Book 2.2, split text into separate *words*
r = [i for i in re.split(r"([,.:;?_!\"()']|--|\s)", raw_text) if i.strip()]


# Book 2.3, create vocabulary   word->idx
# i dont understand why to sorted() the vocabulary, but whatever
vocab = {w: idx for idx, w in enumerate(sorted(set(r)))}
reversed_vocab = {idx: w for w, idx in vocab.items()}

class SimpleTokenizerV1:
    # typehints mine
    def __init__(self, vocab: dict):
        # this clearly misses the "training" part
        # vocab should have default value for UNKOWN tokens  -> 2.4 section
        self.str_to_int = vocab
        self.int_to_str = {i: w for w,i in vocab.items()}

    def encode(self, text: str):
        r = [i for i in re.split(r"([,.:;?_!\"()']|--|\s)", text) if i.strip()]
        return [self.str_to_int[word] for word in r]

    def decode(self, ids: list[int]):
        joined_str = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r"\s+([,.?!\"()'])", r"\1", joined_str)


tokenizer = SimpleTokenizerV1(vocab)

# test the implementation with book example
ids = tokenizer.encode(""""It's the last he painted, you know,"
                        Mrs. Gisburn said with pardonable pride.""")

# print(tokenizer.decode(ids))

# Book 2.4
# in lit I usually found capital text, how so? is lowercase/upercase important?
EOT = "<|endoftext|>"
UNK = "<|unk|>"

# this really should be part of the class
vocab = {w: idx for idx, w in enumerate(sorted(set(r))+[EOT, UNK])}
reversed_vocab = {idx: w for w, idx in vocab.items()}


class SimpleTokenizerV2:
    # typehints mine
    def __init__(self, vocab: dict):
        # this clearly misses the "training" part
        # vocab should have default value for UNKOWN tokens  -> 2.4 section
        self.str_to_int = vocab
        self.int_to_str = {i: w for w,i in vocab.items()}

    def encode(self, text: str):
        r = [i for i in re.split(r"([,.:;?_!\"()']|--|\s)", text) if i.strip()]
        # replace OOD for UNK token
        r = [i if i in self.str_to_int else UNK for i in r]  # merge with next line?

        return [self.str_to_int[word] for word in r]

    def decode(self, ids: list[int]):
        joined_str = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r"\s+([,.?!\"()'])", r"\1", joined_str)

# test from the books
t1 = "Hello, do you like tea?"
t2 = "In the sunlitterraces of the palace."
joined = f" {EOT} ".join([t1, t2])

tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(joined))
# print(tokenizer.decode(tokenizer.encode(joined)))

# BOOK 2.5
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
test_text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."

tokenized = tokenizer.encode(test_text, allowed_special={"<|endoftext|>"})
# pprint(tokenizer.decode(tokenized))

# BOOK 2.6
# data preparation


encoded = tokenizer.encode(raw_text)
# pprint(len(encoded))
enc_sample = encoded[50:]

CONTEXT_SIZE = 4
x, y = enc_sample[:CONTEXT_SIZE], enc_sample[1:CONTEXT_SIZE+1]

for i in range(1, CONTEXT_SIZE+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"{context} ---> {desired}")
    print(f"{tokenizer.decode(context)} ---> {tokenizer.decode([desired])}")

## Effective PyTorch loader
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        
        # txt seems like all the string texts
        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids)-max_length, stride):
            self.input_ids.append(torch.tensor(
                token_ids[i:i+max_length]))   # input chunk 
            self.target_ids.append(torch.tensor(
                token_ids[i+1:i+max_length+1]))     # target chunk

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    return DataLoader(dataset, batch_size=batch_size, 
                            shuffle=shuffle, drop_last=drop_last, 
                            num_workers=num_workers)
    
## Test runs 
dl = create_dataloader_v1(raw_text, 2, 4, 2, False)
data_iter = iter(dl)
pprint(next(data_iter))