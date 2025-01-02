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
x = torch.rand(2, 3, 768)    # 2 examples, 3 token each
print(ffn(x).shape)

# book 4.4 - skip connections 
print("-"*10)
class ExampleDeepNN(nn.Module):
    def __init__(self, layer_s: list[int], use_skips: bool):
        super().__init__()
        self.use_skips = use_skips
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_d, out_d), GeLU()) for in_d, out_d in zip(layer_s[:-1], layer_s[1:])
        ])

    def forward(self, x: torch.Tensor):
        for l in self.layers:
            output = l(x)
            if self.use_skips and x.shape == output.shape:
                x = x + output    # dont use += -> leads to inplace runtime s*~@${#~#}
            else:
                x = output
        return x

layers = [3, 3,  3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
no_skips = ExampleDeepNN(layers, False)
print(no_skips(sample_input))
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if "weight" in name:
            print(f"{name} has grad mean {param.grad.abs().mean().item()}")
print_gradients(no_skips, sample_input)

# now skips
print("-"*10)
torch.manual_seed(123)
yes_skips = ExampleDeepNN(layers, True)
print_gradients(yes_skips, sample_input)

GPT_CONFIG_124M = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "emb_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": .1, 
    "qkv_bias": False,
}

# book 4.5  - transformer block
from attention import MultiHeadAttention

class ReferentialTransformerBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.atn = MultiHeadAttention(
            d_in=c["emb_dim"],
            d_out=c["emb_dim"],
            context_length=c["context_length"],
            dropout=c["drop_rate"],
            num_heads=c["n_heads"],
            qkv_bias=c["qkv_bias"],
        )
        self.ff      = FeedForward(c)
        self.norm1   = LayerNorm(c["emb_dim"])
        self.norm2   = LayerNorm(c["emb_dim"])
        self.dropout = nn.Dropout(c["drop_rate"])
    
    def forward(self, x: torch.Tensor):
        skip = x    # save referention
        x = self.norm1(x)
        x = self.atn(x)
        x = self.dropout(x)
        x = x + skip

        skip = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        
        return x + skip


class TransformerBlock(nn.Module):
    """
        this is different than in the book, because i tried to code it without looking at the solution
    """
    def __init__(self, c):
        super().__init__()

        self.decoder = nn.Sequential(
            LayerNorm(emb_dim=c["emb_dim"]),
            MultiHeadAttention(d_in=c["emb_dim"], d_out=c["emb_dim"],
                               context_length=c["context_length"],
                               dropout=c["drop_rate"], num_heads=c["n_heads"],
                               qkv_bias=c["qkv_bias"]),
            nn.Dropout(c["drop_rate"]),
        )

        self.out_layer = nn.Sequential(
            LayerNorm(emb_dim=c["emb_dim"]),
            FeedForward(c), 
            nn.Dropout(c["drop_rate"]),
        )

    def forward(self, x):       # still prefer batch naming for input var
        x = x + self.decoder(x)
        x = x + self.out_layer(x)

        return x

## test it 
print("-"*20)
torch.manual_seed(123)
x = torch.rand(2, 4, 768)    # two examples, 4 tokens each
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print(f"in shape {x.shape} | out shape {output.shape}")
ref_block = ReferentialTransformerBlock(GPT_CONFIG_124M)
ref_output = ref_block(x)
print(f"in shape {x.shape} | out shape {ref_output.shape}")

# book 4.6  - lets get GPT
print("------4.6------")
class GPTModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        # layer of tokenization
        self.tok_emb = nn.Embedding(c["vocab_size"], c["emb_dim"])   # in, out dims
        # layer of positional embedding
        self.pos_emb = nn.Embedding(c["context_length"], c["emb_dim"]) 
        # dropout
        self.drop_emb = nn.Dropout(c["drop_rate"])
        
        # transformer layerS
        self.trf_blocks = nn.Sequential(
            *[ReferentialTransformerBlock(c) for _ in range(c["n_layers"])]
        )

        self.final_norm = LayerNorm(c["emb_dim"])
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
    
# lets test it
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print(f"innput batch: {batch}")
print(f"output shape: {out.shape}")
print(out)

total_params = sum(p.numel() for p in model.parameters())
print(f"total params: {total_params:,}")   # original GPT2 ties together input token embedding and output layer (same dims), resulting in some speedups

# exercise 4.1
# calc params in transformers an FF parts
ff_params  = sum([sum([p.numel() for p in b.ff.parameters()]) for b in model.trf_blocks])
mha_params = sum([sum([p.numel() for p in b.atn.parameters()]) for b in model.trf_blocks])

print(f"FF: {ff_params:,}, MHA: {mha_params:,}")

# get mem footprint
total_size_bytes = total_params * 4     # float 32 has 4 bytes
total_size_mb = total_size_bytes / (1024*1024)
print(f"total mem: {total_size_mb:.2f}MB")


print("-----size ests-----")
## exercise 4.2
# get sizes for m and xl gpt2 models
GPT_CONFIG_MEDIUM = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "emb_dim": 1024, 
    "n_heads": 16, 
    "n_layers": 24,
    "drop_rate": .1, 
    "qkv_bias": False,
}
# this takes too long
# model_m = GPTModel(GPT_CONFIG_MEDIUM)
# total_params = sum(p.numel() for p in model_m.parameters())
# total_size_mb = (total_params*4) / (1024*1024*1024)
# print(f"total params medium : {total_params:,}")   # original GPT2 ties together input token embedding and output layer (same dims), resulting in some speedups
# print(f"total mem medium: {total_size_mb:.2f}GB")

GPT_CONFIG_LARGE = {
    "vocab_size": 50257, 
    "context_length": 1024, 
    "emb_dim": 1600, 
    "n_heads": 25, 
    "n_layers": 48,
    "drop_rate": .1, 
    "qkv_bias": False,
}

# this takes too long
# model_xl = GPTModel(GPT_CONFIG_LARGE)
# total_params = sum(p.numel() for p in model_xl.parameters())
# total_size_mb = (total_params*4) / (1024*1024*1024)
# print(f"total params large : {total_params:,}")   # original GPT2 ties together input token embedding and output layer (same dims), resulting in some speedups
# print(f"total mem large: {total_size_mb:.2f}GB")

# book 4.7
# inference with softmax and 
print("====4.7====")
def generate_text_simple(model, tokens, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -context_size:]
        with torch.no_grad():
            logits = model(tokens_cond)
        
        logits = logits[:, -1, :]  # last step of every batch part
        probas = torch.softmax(logits, dim=-1)
        token_next = torch.argmax(probas, dim=-1, keepdim=True)
        tokens = torch.cat((tokens, token_next), dim=1)
    
    return tokens

# test it

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print(f"enc: {encoded}")
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print(f"enc tensor shape {encoded_tensor.shape}")

model.eval()   # turns off dropouts and such
out = generate_text_simple(model, encoded_tensor, 6, GPT_CONFIG_124M["context_length"])
print(f"out: {out}")
print(f"out len: {out.shape}")

# ids 2 tokens
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(f"decoded {decoded_text}")
