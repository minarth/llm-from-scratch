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