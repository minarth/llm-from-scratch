import tiktoken
from gpt import generate_text_simple, GPT_CONFIG_124M, GPTModel
import torch

print("###"*10)
print("###"*10)
print("###"*10)

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(ids, tokenizer):
    return tokenizer.decode(ids.squeeze(0).tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

GPT_CONFIG = {
    "vocab_size": 50257, 
    "context_length": 256, 
    "emb_dim": 768, 
    "n_heads": 12, 
    "n_layers": 12,
    "drop_rate": .1, 
    "qkv_bias": False,
}
torch.set_printoptions(sci_mode=True)
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG)
model.eval()

token_ids = generate_text_simple(
    model=model, 
    tokens=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG["context_length"]
)

print(f"OUT: {token_ids_to_text(token_ids, tokenizer)}")

# loss function
inputs  = torch.tensor([[16833, 3626, 6100], [40, 1107, 588]])
targets = torch.tensor([[3626, 6100, 345], [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)

probas = torch.softmax(logits, dim=-1)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print(f"Token ids: {token_ids}")

# compare target vs real outcome
print(f"target 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"output 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

print(f"probas t1: {probas[0, [0, 1, 2], targets[0]]}")
print(f"probas t2: {probas[1, [0, 1, 2], targets[1]]}")

target_probas_1 = probas[0, [0, 1, 2], targets[0]]
target_probas_2 = probas[1, [0, 1, 2], targets[1]]

log_probas = torch.log(torch.cat([target_probas_1, target_probas_2]))
print(log_probas)
avg_log_probas = log_probas.mean()
print(avg_log_probas)
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)

# use cross entropy loss 
print(f"logits shape: {logits.shape}")
print(f"targets. shape : {targets.shape}")

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
print(f"l shape {logits_flat.shape} t shape {targets_flat.shape}")

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)

# read the data
file_path = "data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as fd:
    text_data = fd.read()

print(f"chars: {len(text_data)} tokens: {len(tokenizer.encode(text_data))}")


# lets get to the prep of data 
train_ratio = .9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

from tokenization import create_dataloader_v1

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=True,
    shuffle=True, 
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG["context_length"],
    stride=GPT_CONFIG["context_length"],
    drop_last=False,
    shuffle=False, 
    num_workers=0,
)

print("train loader:")
for x,y in train_loader:
    print(x.shape, y.shape)

print("val loader:")
for x,y in val_loader:
    print(x.shape, y.shape)

def batch_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)     # move to gpus
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_batch.flatten()
        )
    
    return loss


def loader_loss(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    
    total_loss = .0
    # this structure looks ugly
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (in_batch, tgt_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = batch_loss(in_batch, tgt_batch, model, device)
        total_loss += loss
    
    return total_loss / num_batches


print(torch.cuda.is_available())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

with torch.no_grad():
    train_loss = loader_loss(train_loader, model, device)
    val_loss   = loader_loss(val_loader, model, device)

print(f"TRN loss: {train_loss}\nVAL loss: {val_loss}")


################################################################################

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_l = loader_loss(train_loader, model, device, num_batches=eval_iter)
        val_l   = loader_loss(val_loader, model, device, num_batches=eval_iter)
    model.train()  # reverse model.eval() setup
    return train_l, val_l


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model, encoded, max_new_tokens=50, 
                                         context_size=context_size)
    decoded = token_ids_to_text(token_ids, tokenizer) 
    print(decoded.replace('\n', ' EOL '))

    model.train()


# training, types for this are hard vv
def train_model_simple(
        model,
        train_loader, 
        val_loader,
        optimizer, 
        device, 
        num_epochs, 
        eval_freq, 
        eval_iter,          # same as num_batches in loss evaluation for whole data loader, just, why? 
        start_context, 
        tokenizer
):
    
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()    # main train loop
        #for input_batch, target_batch in train_loader:
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(x_batch, y_batch, model, device)
            loss.backward()   # gradient calc, its like magic
            optimizer.step()

            tokens_seen += x_batch.numel()   # number of elements in structure https://pytorch.org/docs/stable/generated/torch.numel.html
            global_step += 1

            if global_step % eval_freq == 0: 
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                track_tokens_seen.append(tokens_seen)

                print(f"EP {epoch+1} step {global_step:06d}: trainL {train_loss:.3f}, valL {val_loss:.3f}")

        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen



#######
# run small train loop
#######

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG)    # smaller config
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=.0004,
    weight_decay=.1
)

NUM_EPOCHS = 10
EVAL_FREQ = 5
EVAL_ITER = 5
START_CONTEXT = "Every effort moves you"
print("STARTING TRAINING LOOP")


import matplotlib.pyplot as plt
def plot_losses(epochs: list[int], tokens: list[int], train_l: list[float], val_l: list[float]):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs, train_l, label="train")
    ax1.plot(epochs, val_l, label="val", linestyle="-.")
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")

    ax1.legend(loc="upper right")

    ax2 = ax1.twiny()
    ax2.plot(tokens, train_l, alpha=0)
    ax2.set_xlabel("tokens")
    fig.tight_layout()


    plt.savefig("gpt2_train_v_val.png")

if False:
    train_l, val_l, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer,
        device, NUM_EPOCHS, EVAL_FREQ, EVAL_ITER, START_CONTEXT, tokenizer
    )

    torch.save(model, "data/gpt2-smol")

    # show the train loss and validation loss next to each other
    epoch_tensor = torch.linspace(1, NUM_EPOCHS, len(train_l))
    plot_losses(epoch_tensor, tokens_seen, train_l, val_l)
else:
    model = torch.load("data/gpt2-smol", weights_only=False )
    model.eval()


## lets go
# book 5.3 - random decoding

model.to("cpu")
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(model, 
                                 text_to_token_ids("Every effort moves you", tokenizer), 
                                 max_new_tokens=25, 
                                 context_size=GPT_CONFIG["context_length"])

print(f"output text: {token_ids_to_text(token_ids, tokenizer)}")

# temp scaling

vocab = {"closer": 0, "every": 1, "effort":2, "forward": 3, "inches": 4,
         "moves": 5, "pizza": 6, "toward": 7, "you": 8}
inverse_vocab = {v: k for k, v in vocab.items()}

# random example
next_token_logits = torch.tensor([
    4.51, .89, -1.9, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79
])

probas = torch.softmax(next_token_logits, dim=0)
next_token_idx = torch.argmax(probas).item()
print(probas)
print(inverse_vocab[next_token_idx])

torch.manual_seed(123)
# now do multinomial sample https://pytorch.org/docs/stable/generated/torch.multinomial.html
next_token_id = torch.multinomial(probas, num_samples=1).item() # item -> returns value from the tensor
print(f"sample: {inverse_vocab[next_token_id]}")

# lets do 1000 samples
def sample_n(probas, n):
    torch.manual_seed(123)
    samples = [torch.multinomial(probas, num_samples=1).item() for _ in range(n)]
    ids = torch.bincount(torch.tensor(samples))
    for i, freq in enumerate(ids):
        print(f"{inverse_vocab[i]}: {freq}")

sample_n(probas, 1000)

# i had to disable cursor suggestions cause learning too little
# temperature scaling

# temperature is just fancy word for dividing logits -> (0, 1> peaky distr, (1, inf) unfirm-ish distr
def softmax_with_temp(logits, temperature: float) -> torch.Tensor:
    assert temperature > 0, "temp needs to be positive!"
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

temps = [1, .1, 5, ]

scaled_probas = [softmax_with_temp(next_token_logits, t) for t in temps]

x = torch.arange(len(vocab))

bar_width = .15
fig, ax = plt.subplots(figsize=(5,3))
for i, t in enumerate(temps):
    rects = ax.bar(x + i*bar_width, scaled_probas[i],
                   bar_width, label=f"t {t}")
    ax.set_ylabel("probs")
    ax.set_xticks(x)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig("temp.png")    


# exercise 5.1, print sample freqs
print("%"*10)
for i, t in enumerate(temps):
    print(t)
    print(f"probs: {scaled_probas[i]}")
    sample_n(scaled_probas[i], 1000)
    print("-"*10)

# topk sampling
## logits -> top3 -> -inf mask -> softmax
TOP_K = 3
top_logits, top_pos = torch.topk(next_token_logits, k=TOP_K)
print(f"top lgst: {top_logits}")
print(f"top positions: {top_pos}")

### -inf mask
masked_logits = torch.where(
    condition=next_token_logits < top_logits[-1],        # take the lowest from TOP
    input=torch.tensor(float("-inf")),
    other=next_token_logits
)
print(masked_logits)

### softmax
top_probas = torch.softmax(masked_logits, dim=0)
print(f"probas: {top_probas}")

def generate(model, tokens, max_new_tokens, context_size, 
             temperature=.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        tokens_cond = tokens[:, -context_size:]
        with torch.no_grad():
            logits = model(tokens_cond)
        
        logits = logits[:, -1, :]  # last step of every batch part
        if top_k:
            top_logits, _ = torch.topk(logits, k=top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float("-inf")),
                other=logits,
            )
        if temperature > .0:
            logits = logits / temperature
            probas = torch.softmax(logits, dim=-1)
            token_next = torch.multinomial(probas, num_samples=1)
        else:
            token_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        if token_next == eos_id:
            break
        tokens = torch.cat((tokens, token_next), dim=1)
    
    return tokens

torch.manual_seed(123)
token_ids = generate(
    model, 
    text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15, 
    context_size=GPT_CONFIG["context_length"],
    top_k=25,
    temperature=1.4,
)

print(f"topk + temp: {token_ids_to_text(token_ids, tokenizer)}")

# book 5.4 - save/load... already doing that so the speed of run is good (above)
torch.save(model.state_dict(), "data/model.pth")

model2 = GPTModel(GPT_CONFIG)
model2.load_state_dict(torch.load("data/model.pth", map_location=device))
model2.eval() # switch to eval mode

# smarter save (w optimizer)

torch.save({
    "model_state_dict": model.state_dict(), 
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "data/model_n_optimizer.pth"
    )

checkpoint = torch.load("data/model_n_optimizer.pth", map_location=device)
model3 = GPTModel(GPT_CONFIG)
model3.load_state_dict(checkpoint["model_state_dict"])

optimizer2 = torch.optim.AdamW(model3.parameters(), lr=5e-4, weight_decay=.1)
optimizer2.load_state_dict(checkpoint["optimizer_state_dict"])

# exercise 5.4
print("Additional training for loaded model+opt")
model3.train()
tl, vl, tokens = train_model_simple(
    model3,
    train_loader, 
    val_loader, 
    optimizer2,
    device,
    num_epochs=0, # change to 1 to start training
    eval_freq=EVAL_FREQ, 
    eval_iter=EVAL_ITER,
    start_context=START_CONTEXT,
    tokenizer=tokenizer
)

# book 5.5 load openAI weights

from gpt_download import download_and_load_gpt2
settings, params = download_and_load_gpt2(
    model_size="124M",
    models_dir="gpt2"
)

print(f"set: {settings}")
print(f"params: {params.keys()}")

print(params["wte"])
print(f"tkn emb shape: {params['wte'].shape}")

model_configs = {
    "gpt-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt-small"
OPENAI_CONFIG = GPT_CONFIG.copy()
OPENAI_CONFIG.update(model_configs[model_name])

OPENAI_CONFIG.update({"context_length": 1024})
OPENAI_CONFIG.update({"qkv_bias": True})

gpt2 = GPTModel(OPENAI_CONFIG)
gpt2.eval()

def assign(left: torch.Tensor, right: torch.Tensor):
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch {left.shape} != {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

import numpy as np
def load_weights_into_gpt(gpt: GPTModel, params):
    # this is a lot of code, had to copy from github
    ## https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/01_main-chapter-code/ch05.ipynb

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        trf = gpt.trf_blocks[b]  # transformer
        atn = trf.atn
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].atn.W_query.weight = assign(
            gpt.trf_blocks[b].atn.W_query.weight, q_w.T)
        gpt.trf_blocks[b].atn.W_key.weight = assign(
            gpt.trf_blocks[b].atn.W_key.weight, k_w.T)
        gpt.trf_blocks[b].atn.W_value.weight = assign(
            gpt.trf_blocks[b].atn.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].atn.W_query.bias = assign(
            gpt.trf_blocks[b].atn.W_query.bias, q_b)
        gpt.trf_blocks[b].atn.W_key.bias = assign(
            gpt.trf_blocks[b].atn.W_key.bias, k_b)
        gpt.trf_blocks[b].atn.W_value.bias = assign(
            gpt.trf_blocks[b].atn.W_value.bias, v_b)

        gpt.trf_blocks[b].atn.out_projection.weight = assign(
            gpt.trf_blocks[b].atn.out_projection.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].atn.out_projection.bias = assign(
            gpt.trf_blocks[b].atn.out_projection.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

load_weights_into_gpt(gpt2, params)
gpt2.to(device)

torch.manual_seed(123)
token_ids = generate(
    model=gpt2,
    tokens=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=OPENAI_CONFIG["context_length"],
    top_k=50,
    temperature=1.5 # -> more uniformed
)
print(f"Output from GPT2: {token_ids_to_text(token_ids, tokenizer)}")

# exercise 5.5
## evaluate the gpt2 model and calc the losses

gpt_tl, gpt_vl = evaluate_model(gpt2, train_loader, val_loader, device, None)
print(f"GPT train loss: {gpt_tl} and val loss: {gpt_vl}")
