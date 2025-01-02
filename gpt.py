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

# BOOK 4.2  - layer norm 
print("="*20)
torch.manual_seed(123)
torch.set_printoptions(sci_mode=False)    # optional, just better/different output
batch_example = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())  
# nn.Sequential is a container that provides sequential processing of ordered layers
out = layer(batch_example)
print(out)
mean = out.mean(dim=-1, keepdim=True)
print(f"mean: {mean}")
var = out.var(dim=-1, keepdim=True)
print(f"variance: {var}")
print("----NORMED----")
out_norm = (out-mean)/torch.sqrt(var)    # standard deviation
print(out_norm)
mean = out_norm.mean(dim=-1, keepdim=True)
print(f"mean: {mean}")
var = out_norm.var(dim=-1, keepdim=True)
print(f"variance: {var}")

class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int, eps: float=1e-5):
        super().__init__()
        self.eps = eps

        # creating scale and shift params if training finds better mean, var to aim for
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)    # keepdim important for further application
        variance = x.var(dim=-1, keepdim=True, unbiased=False)  # this is BIASED variance, not using n-1 to calc n
        normed = (x-mean) / torch.sqrt(variance+self.eps)
        
        return self.scale * normed + self.shift
    
# test it

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print(f"m: {mean}, v: {var}")


# BOOK 4.3
# gelu (gaussian error linear unit) implementation

class GeLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return .5*x*(1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) 
                     * (x + .044715 * torch.pow(x,3))))   # pow vs ** ?

from term_plot import plot_to_terminal

import matplotlib.pyplot as plt
gelu, relu = GeLU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
fig = plt.figure(figsize=(8,3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GeLU", "ReLU"]), 1):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f"{label} act fnc")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.savefig("gelu_v_relu2.png")
plt.close()


class FeedForward(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(c["emb_dim"], 4*c["emb_dim"]),    
            GeLU(),
            nn.Linear(4*c["emb_dim"], c["emb_dim"])      # why the expansion and contraction? 
        )
    def forward(self, x):
        return self.layers(x)

ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
print(ffn(x).shape)
