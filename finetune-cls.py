# book 6
## classic classification - spam/ham

# book 6.2 dwnld data
import pandas as pd
import urllib.request
import zipfile, os
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "data/sms_spam_collection.zip"
EXTRACTED_PATH = "data/sms_spam_collection"
DATA_FILE_PATH = Path(EXTRACTED_PATH) / "SMSSpamCollection.tsv"

def download_and_unzip_spam_data(
        url, zip_p, extracted_p, data_file_p
):
    if data_file_p.exists():
        print(f"everything dwnldd, skip")
        return
    
    # save url docs to zip path
    with urllib.request.urlopen(url) as response:
        with open(zip_p, "wb") as out_fd:
            out_fd.write(response.read())
    
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(extracted_p)
    
    original_fp = Path(extracted_p) / "SMSSpamCollection"
    os.rename(original_fp, data_file_p)
    print(f"dwnlded and saved as {data_file_p}")

download_and_unzip_spam_data(URL, ZIP_PATH, EXTRACTED_PATH, DATA_FILE_PATH)


df = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
print(df["Label"])

def create_balance_df(df: pd.DataFrame) -> pd.DataFrame:
    lower_class_num = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(lower_class_num, random_state=123)

    return pd.concat(
        [ham_subset, df[df["Label"] == "spam"]]
    )
balanced_df = create_balance_df(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# now random split 70, 10, 20 (train, test, val)

def random_split(df: pd.DataFrame, train_fr: float, val_fr: float) -> list[pd.DataFrame]:
    df = df.sample(frac=1., random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_fr)
    val_end   = train_end + int(len(df) * val_fr)
    
    # train, val, test
    return df[:train_end], df[train_end:val_end], df[:val_end]

train_df, val_df, test_df = random_split(balanced_df, .7, .2)

# save it to be sure
train_df.to_csv("data/train.csv", index=None)
val_df.to_csv("data/validation.csv", index=None)
test_df.to_csv("data/test.csv", index=None)


# book 6.3
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: tiktoken.Encoding, max_len: int=None,
                 pad_token_id: int=50256):
        super().__init__()

        self.data = pd.read_csv(csv_file)
        self.encoded = [tokenizer.encode(text) for text in self.data["Text"]]

        if not max_len:
            self.max_len = self._longest_encoded()
        else:
            self.max_len = max_len
            # truncate longer seqs than entered max
            self.encoded = [enc[:self.max_len] for enc in self.encoded]

        self.encoded = [
            enc + [pad_token_id]*(self.max_len - len(enc))
            for enc in self.encoded
        ]

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        enc = self.encoded[index]
        label = self.data.iloc[index]["Label"]

        return (
            torch.tensor(enc, dtype=torch.long),
            torch.tensor(label, dtype=torch.long), 
        )
    
    def __len__(self) -> int:
        return len(self.encoded)

    def _longest_encoded(self) -> int:
        return max([len(e) for e in self.encoded])

train_ds = SpamDataset("data/train.csv", tokenizer) 
print(f"max len in train: {train_ds.max_len} loaded samples {len(train_ds)}")

val_ds = SpamDataset("data/validation.csv", tokenizer, train_ds.max_len)
test_ds = SpamDataset("data/test.csv", tokenizer, train_ds.max_len)

from torch.utils.data import DataLoader
NUM_WORKERS = 0
BATCH_SIZE  = 8

torch.manual_seed(123)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
)

input_batch, target_batch = next(iter(train_loader))   # cleaner than for .. in
print(f"inp b dims {input_batch.shape}")
print(f"target b dim {target_batch.shape}")


print("-"*10)
print(f"train b {len(train_loader)}, test b {len(test_loader)}, val b {len(val_loader)}")

# book 6.4 init the model
## gpt2 S again

#### load the model

MODEL_NM = "gpt2-small"
MODEL_SIZE = {
    "gpt2-small": "124M",
}
INPUT_PROMPT = "Every effort moves" # he needs to get more creative

# i have the configs defined in training.py, duplicating it so it is not interdependent
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": .0,
    "qkv_bias": True,
}

OPENAI_MODEL = {
    "gpt2-small": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

BASE_CONFIG.update(OPENAI_MODEL[MODEL_NM])

from gpt_download import download_and_load_gpt2
from gpt import GPTModel
from training import load_weights_into_gpt

settings, params = download_and_load_gpt2(model_size=MODEL_SIZE[MODEL_NM],
                                          models_dir="gpt2")
model = GPTModel(BASE_CONFIG)   # dont like the naming
load_weights_into_gpt(model, params)
model.eval()

# test generation
from gpt import generate_text_simple
from training import text_to_token_ids, token_ids_to_text
print("="*20)
text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model, text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG["context_length"]
)

print(f"test string: {token_ids_to_text(token_ids, tokenizer)}")

text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or $2000 award.'"
)

token_ids = generate_text_simple(
    model,
    text_to_token_ids(text_2, tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG["context_length"],
)
print(f"instruct spam detection: {token_ids_to_text(token_ids, tokenizer)}")

# book 6.5
# print(model)  # prints nice output

## lets freeze the layers
for param in model.parameters():
    param.requires_grad = False

## replace end layer
torch.manual_seed(123)
NUM_CLASSES = 2
model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG["emb_dim"],
    out_features=NUM_CLASSES
)

# to improve finetune perf
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True

## test calling it with text
