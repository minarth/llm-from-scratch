GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "emb_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": .1, 
    "qkv_bias": False,
}

# BOOK 4.1
# now Dummy-GPTModel
import torch
import torch.nn as nn

class DummyLayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps=1e-5):
        super().__init__()
        self.shape = emb_dim

    def forward(self, x):
        return x
        

class DummyTransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()

    def forward(self, x):
        return x


class DummyGPTModel(nn.Module):
    """
        tokenized text
            v
        embeddin layers   (token emb + positional emb)
            v
        transformer  (mutli head attn, normalization, shortcuts)
            v
        output layer
    """
    def __init__(self, c):
        #  c a.k.a config
        super().__init__()

        # layer of tokenization
        self.tok_emb = nn.Embedding(c["vocab_size"], c["emb_dim"])   # in, out dims
        # layer of positional embedding
        self.pos_emb = nn.Embedding(c["context_length"], c["emb_dim"]) 
        # dropout
        self.drop_emb = nn.Dropout(c["drop_rate"])
        
        # transformer layerS
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(c) for _ in range(c["n_layers"])]
        )

        self.final_norm = DummyLayerNorm(c["emb_dim"])
        self.out_head   = nn.Linear(c["emb_dim"], c["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, sequence_len = in_idx.shape
        tok_e = self.tok_emb(in_idx)
        
        # this retrives vectors for positions in sequence 
        #   (first token in sequence has always same pos vector)
        pos_e = self.pos_emb(torch.arange(sequence_len, device=in_idx.device))

        x = tok_e + pos_e    
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        return logits


# data prep example

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1  = "Every effort moves you"
txt2  = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)
print(f"shape of batch: {batch.shape}")
print(f"batch: {batch}")
print("-"*10)
# init model
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)

print(f"output shape {logits.shape}\nlogits {logits}")
