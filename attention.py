
##### BOOK 3
### Attention https://giphy.com/gifs/up-attention-alert-BafZrA7tQuk4x997G0

import torch

# Bahdanau attention, 2014

# Book 3.3, simple Self-Attention
# output od SA is called "context vector"

# sample embedded input
inputs = torch.tensor([
    [.43, .15, .89],    # Your      x^1
    [.55, .87, .66],    # journey   x^2
    [.57, .85, .64],    # starts    x^3
    [.22, .58, .33],    # with      x^4
    [.77, .25, .1 ],    # first     x^5
    [.05, .8,  .55],    # step      x^6
])

# we now calc attention for input x^2 "journey" <- called QUERY token
# calc attention scores omega (intermediate values)
#   dot products between every element and the selected query token

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])   # scores for query token 2, initialize for iterative calc
for i, x_i in enumerate(inputs): 
    attn_scores_2[i] = torch.dot(query, x_i)
print(attn_scores_2)

# How I would do it, but not that clear what is happening
# print(torch.matmul(inputs, query))

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()   # normalize so sum()=1
# in realworld softmax is advised for normalization

print(f"[Classic] Weights: {attn_weights_2_tmp} and sum = {attn_weights_2_tmp.sum()}")

def softmax_naive(x):
    # there is a torch function for this, but this is more clear
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print(f"[Softmax naive] Weights: {attn_weights_2_naive} and sum = {attn_weights_2_naive.sum()}")

#how it should be
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print(f"[Softmax ref] Weights: {attn_weights_2} and sum = {attn_weights_2.sum()}")

# calculating context vector for x^2
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i     # x_i == inputs[i]
print(f"Context V 2: {context_vec_2}")

## generalize for whole input
attn_scores = inputs @ inputs.T                     # dot products between every row vector
attn_weights = torch.softmax(attn_scores, dim=1)   # normalize so each row sums to 1;   dim 0-over rows(vertical dir), 1-over cols(horizontal dir), -1-over last one
print(attn_weights)

# Torch trivia shape -> [num of rows, num of cols], -1 -> over cols
context_vectors = attn_weights @ inputs   # making error in dim=X can go unnoticed
print(context_vectors)

print(f"Sanity check {context_vec_2} == {context_vectors[1]}")

# Book 3.4 - trainable weights a.k.a. scaled dot-product attention
# again, starting with second input as a selected query input

x_2 = inputs[1]    # query input, but "query" has different meaning now, so calling it x_2

DIM_IN = inputs.shape[1]    # number of cols
DIM_OUT = 2                 # selected, usually same as input one, 2 for EDU purposes

# we do this for one attn head
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(DIM_IN, DIM_OUT), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(DIM_IN, DIM_OUT), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(DIM_IN, DIM_OUT), requires_grad=False)
# requires_grad=False for printable outputs, will set to True in later chapters

query_2 = x_2 @ W_query     # we calc query only for query token, rest is against all inputs
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

keys   = inputs @ W_key
values = inputs @ W_value
print(f"Keys shape {keys.shape}")
print(f"Values shape {values.shape}")

# simple calc for one instance
attn_score_22 = torch.dot(query_2, key_2)
print(attn_score_22)

# full matmul calc
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# normalization sqrt(d_k), called scaled-dot -> improves training perf
D_K = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / D_K ** 0.5, dim=-1)
print(attn_weights_2)

# calc context vectors
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


# compact implementation
import torch.nn as nn
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys    = x @ self.W_key
        queries = x @ self.W_query
        values  = x @ self.W_value

        attn_scores  = queries @ keys.T   # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)      # -1 dims are making me uneasy
        context_vec = attn_weights @ values
        # print(f"Shapes: scores {attn_scores.shape}, weights {attn_weights.shape}, context {context_vec.shape}")
        return context_vec
    

# SELF ATTENTION V1
torch.manual_seed(123)
sa_v1 = SelfAttentionV1(DIM_IN, DIM_OUT)
#print(sa_v1(inputs))

# Self Attention V2 - using nn.Linear
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        queries = self.W_query(x)
        keys    = self.W_key(x)
        values  = self.W_value(x)

        attn_scores  = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) 
        return attn_weights @ values
    
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(DIM_IN, DIM_OUT)


# excercise 3.1, transfering W_i weights
print("exc")
print(sa_v2.W_query.weight)
print(sa_v1.W_query.data)

sa_v1.W_query.data = sa_v2.W_query.weight.T
sa_v1.W_key.data   = sa_v2.W_key.weight.T
sa_v1.W_value.data = sa_v2.W_value.weight.T

print("TEST")

print(f"V2: {sa_v2(inputs)}")
print(f"V1: {sa_v1(inputs)}")

# BOOK 3.5
# Causal ~ Masked attention

# tired of writing
sa = sa_v2
norm = lambda s, k: torch.softmax(s / k.shape[-1]**.5, dim=-1)  #scaled dot norm


q = sa.W_query(inputs)
k = sa.W_key(inputs)
attn_scores = q @ k.T
attn_weights = norm(attn_scores, k)
print(attn_weights)

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))    # tril -> returns lower triangular
print(mask_simple)

masked_simple = attn_weights * mask_simple   # * –> element-wise multiplication
print(masked_simple)

# now simple renormalization
masked_simple_norm = masked_simple / masked_simple.sum(dim=-1, keepdim=True)
print(masked_simple_norm, masked_simple_norm.sum(dim=1))  # all good

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)  # fill True positions with -inf
#print(masked)
attn_weights = norm(masked, k)
print(attn_weights) 

# add dropout
torch.manual_seed(123)
dropout = torch.nn.Dropout(.5)
 
print(dropout(attn_weights))

#####

class CausalAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, 
                 dropout: float, qkv_bias: bool = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        
        self.context_length = context_length  # unused?
        self.d_out = d_out

        self.register_buffer("mask", 
                             torch.triu(torch.ones(context_length, context_length),
                                        diagonal=1)
                            )    # creating self.mask buffer https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer

    def _norm(self, s, k):
        #scaled dot 
        return torch.softmax(s / k.shape[-1]**.5, dim=-1)

    def forward(self, x):
        # x can be batch
        b, num_tokens, d_in = x.shape
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        attn_scores = q @ k.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # ends with _ -> inplace op
        
        attn_w = self._norm(attn_scores, k)        
        attn_w = self.dropout(attn_w)

        context_vec = attn_w @ v
        return context_vec

print("Causal Attn impl test")
torch.manual_seed(123)
# lets work on batches
batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
ca = CausalAttention(DIM_IN, DIM_OUT, context_length, .0)
context_v = ca(batch)

print(f"Shape: {context_v.shape}")

# book 3.6    -  multihead

#simple wrapping
class MultiHeadAttnWrapper(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.heads = nn.ModuleList(         # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) 
             for _ in range(num_heads)]
        )
    def forward(self, batch):
        return torch.cat([h(batch) for h in self.heads], dim=-1)            # -1 is effective when moving from single matrix to batch inputs

## test MH attn
torch.manual_seed(123)    # only for keeping step with the book
cl = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttnWrapper(d_in, d_out, cl, .0, num_heads=2)
cont_vec = mha(batch)
print(cont_vec)
print(f"Shape: {cont_vec.shape}")

# exercise 3.2
# torch cat concatenates two head outputs [2,6,2] into [2,6,4].. d_out = 1 should make the trick
mha = MultiHeadAttnWrapper(d_in, 1, cl, .0, num_heads=2)
cont_vec = mha(batch)
print(cont_vec)
print(f"Shape: {cont_vec.shape}")


# parallel multihead attn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out muset be divisible by num_heads"    # nice

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        # MHA building blocks
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", 
                             torch.triu(torch.ones(context_length, context_length),
                                        diagonal=1)
                            )
        self.out_projection = nn.Linear(d_out, d_out)        

    def forward(self, batch):
        # using batch name to signal we are passed single "x" input
        b, num_tokens, d_in = batch.shape
        
        # i like to use q,k,v but somehow it make the readibility worse
        queries = self.W_query(batch)
        keys    = self.W_key(batch)
        values  = self.W_value(batch)
        #    -> all shape = b, num_tokens, d_out

        # now reshape/split the matricies to separate heads
        #   (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # now transpose, we need to work with num_tokens x head_dim matricies
        #   (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1,2)
        keys    = keys.transpose(1,2)
        values  = values.transpose(1,2)

        attn_score = queries @ keys.transpose(2,3)
        mask_bool  = self.mask.bool()[:num_tokens, :num_tokens]
        attn_score.masked_fill_(mask_bool, -torch.inf)   # inplace 

        attn_w = torch.softmax(attn_score / keys.shape[-1] ** .5, dim=-1)
        attn_w = self.dropout(attn_w)

        context_v =  (attn_w @ values).transpose(1,2)    # transpose to b, num_tok, n_head, head_dim  -> back to original shape after view()

        # https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        context_v = context_v.contiguous().view(b, num_tokens, self.d_out)
        context_v = self.out_projection(context_v)   # combination in linlayer

        return context_v


## transposition madness ilustrated
print("="*30)
print("MHA transposition madness")

# dim (b,num_heads,num_tokens,head_dim) = [1,2,3,4]
a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
print(a)

# print()
# print(a @ a.transpose(2,3))
# first_head = a[0, 0, :, :]

# print(f"1st head {first_head}")
# first_res = first_head @ first_head.T
# print(f"first res {first_res}")

print(a.view(1, 2, 12))
print("="*30)
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, .0, num_heads=2)
context_vec = mha(batch)
print(context_vec)
print(context_vec.shape)

# exercise 3.3
gpt2_mha = MultiHeadAttention(d_in=768, 
                             d_out=768, 
                             context_length=1024, 
                             dropout=0.1, 
                             num_heads=12)