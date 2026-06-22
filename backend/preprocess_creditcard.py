import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/raw/creditcard.csv"
PROCESSED_DIR = "data/processed"

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")
SAMPLE_TEST_PATH = "data/sample_test.csv"
SAMPLE_TEST_LABELED_PATH = "data/sample_test_labeled.csv"


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    if "Class" not in df.columns:
        raise ValueError("Expected target column 'Class' was not found.")

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["Class"]
    )

    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Fit scaler on train only, then transform both train and test.
    train_df["Amount"] = scaler.fit_transform(train_df[["Amount"]])
    test_df["Amount"] = scaler.transform(test_df[["Amount"]])

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # Balanced demo sample so the UI usually shows both classes.
    legit = test_df[test_df["Class"] == 0].sample(n=180, random_state=42)
    fraud = test_df[test_df["Class"] == 1].sample(n=20, random_state=42)
    sample_labeled = (
        pd.concat([legit, fraud])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    sample_unlabeled = sample_labeled.drop(columns=["Class"])

    sample_unlabeled.to_csv(SAMPLE_TEST_PATH, index=False)
    sample_labeled.to_csv(SAMPLE_TEST_LABELED_PATH, index=False)

    print("Preprocessing complete.")
    print(f"Saved: {TRAIN_PATH}")
    print(f"Saved: {TEST_PATH}")
    print(f"Saved: {SAMPLE_TEST_PATH}")
    print(f"Saved: {SAMPLE_TEST_LABELED_PATH}")
    print("Train class counts:")
    print(train_df["Class"].value_counts().to_dict())
    print("Sample class counts:")
    print(sample_labeled["Class"].value_counts().to_dict())


if __name__ == "__main__":
    main()
