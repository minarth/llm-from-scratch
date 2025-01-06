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
train_l, val_l, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer,
    device, NUM_EPOCHS, EVAL_FREQ, EVAL_ITER, START_CONTEXT, tokenizer
)

