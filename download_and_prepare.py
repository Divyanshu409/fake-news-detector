import os
import pandas as pd
import subprocess
from sklearn.model_selection import train_test_split
import sys
sys.stdout.reconfigure(encoding="utf-8")
# Configuration
DATA_DIR = "data"
DATASET_ID = "clmentbisaillon/fake-and-real-news-dataset"
PROCESSED_FILE = os.path.join(DATA_DIR, "processed_data.csv")

TEST_SIZE = 0.2
RANDOM_SEED = 42
MAX_SAMPLES = 20000


def download_kaggle_dataset():

    print("Downloading dataset from Kaggle...")
    print(f"Dataset: {DATASET_ID}")

    try:
        subprocess.run(
            ["python", "-m", "kaggle", "--version"],
            capture_output=True,
            check=True
        )
    except Exception:
        print("\nKaggle CLI not found!")
        return False

    result = subprocess.run(
        [
            "python",
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET_ID,
            "-p",
            DATA_DIR,
            "--unzip"
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Download failed:")
        print(result.stderr)
        return False

    print("Download complete!\n")
    return True


def load_raw_data():

    true_path = os.path.join(DATA_DIR, "True.csv")
    fake_path = os.path.join(DATA_DIR, "Fake.csv")

    if not os.path.exists(true_path) or not os.path.exists(fake_path):
        print("True.csv or Fake.csv not found.")
        return None

    print("Loading raw data files...")

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)

    df_true["label"] = 1
    df_fake["label"] = 0

    print(f"Real news articles : {len(df_true)}")
    print(f"Fake news articles : {len(df_fake)}")

    return df_true, df_fake


def preprocess(df_true, df_fake):

    print("\nPreprocessing data...")

    df = pd.concat([df_true, df_fake], ignore_index=True)

    df["content"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    )

    df = df[["content", "label"]]

    df = df[df["content"].str.len() > 50]
    df = df.drop_duplicates(subset="content")
    df = df.dropna()

    print(f"Total articles after cleaning : {len(df)}")
    print(f"Real : {(df['label'] == 1).sum()}")
    print(f"Fake : {(df['label'] == 0).sum()}")

    if len(df) > MAX_SAMPLES:

        samples_per_class = MAX_SAMPLES // 2

        real_df = df[df["label"] == 1].sample(
            n=min(samples_per_class, (df["label"] == 1).sum()),
            random_state=RANDOM_SEED
        )

        fake_df = df[df["label"] == 0].sample(
            n=min(samples_per_class, (df["label"] == 0).sum()),
            random_state=RANDOM_SEED
        )

        df = pd.concat([real_df, fake_df], ignore_index=True)

        print(f"Sampled to {len(df)} articles")

    df = df.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print("\nColumns:")
    print(df.columns.tolist())

    return df


def split_and_save(df):

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    df.to_csv(PROCESSED_FILE, index=False)

    train_df.to_csv(
        os.path.join(DATA_DIR, "train.csv"),
        index=False
    )

    test_df.to_csv(
        os.path.join(DATA_DIR, "test.csv"),
        index=False
    )

    print("\nData saved successfully!")
    print(f"Full dataset : {len(df)} rows")
    print(f"Train dataset: {len(train_df)} rows")
    print(f"Test dataset : {len(test_df)} rows")


def main():

    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(PROCESSED_FILE):
        print("processed_data.csv already exists.")
        print("Delete it if you want to regenerate.")
        return

    success = download_kaggle_dataset()

    if not success:
        print("Continuing with local files...")

    result = load_raw_data()

    if result is None:
        return

    df_true, df_fake = result

    df = preprocess(df_true, df_fake)

    split_and_save(df)

    print("\nData preparation complete!")
    print("Run train.py next.")


if __name__ == "__main__":
    main()