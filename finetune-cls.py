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

def create_balance_df(df: pd.DataFrame) -> pd.DataFrame:
    lower_class_num = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(lower_class_num, random_state=123)

    return pd.concat(
        [ham_subset, df[df["Label"] == "spam"]]
    )
balanced_df = create_balance_df(df)["Label"].value_counts()
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# now random split 70, 10, 20 (train, test, val)

def random_split(df: pd.DataFrame, train_fr: float, val_fr: float) -> list[pd.DataFrame]:
    df = df.sample(frac=1., random_state=123).reset_index(drop=True)

    train_end = int(len(df) * train_fr)
    val_end   = train_end + int(len(df) * val_fr)
    
    # train, val, test
    return df[:train_end], df[train_end:val_end], df[val_end]

train_df, val_df, test_df = random_split(balanced_df, .7, .2)

# save it to be sure
train_df.to_csv("data/train.csv", index=None)
val_df.to_csv("data/validation.csv", index=None)
test_df.to_csv("data/test.csv", index=None)


# book 6.3