
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