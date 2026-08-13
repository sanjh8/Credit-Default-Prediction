

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/credit_card_default.xls"
TARGET = "DEFAULT"


def load_raw(path=RAW_PATH):
    # The UCI file has a title row above the real header, so header=1
    df = pd.read_excel(path, header=1)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": TARGET})
    return df


def clean(df):
    df = df.copy()

    # EDUCATION: documented values are 1-4, but 0/5/6 show up as undocumented
    # codes in the raw data -- group them into a single "other" category (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: 0 is an undocumented code -- fold it into "other" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # No missing values in this dataset, but keep this here in case your
    # copy differs / you swap datasets later
    df = df.dropna()

    return df


def main():
    df = load_raw()
    print(f"Loaded raw data: {df.shape}")

    df = clean(df)
    print(f"After cleaning: {df.shape}")
    print(f"Default rate: {df[TARGET].mean():.3%}")

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df[TARGET], random_state=42
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print(f"Saved data/train.csv ({train_df.shape}) and data/test.csv ({test_df.shape})")


if __name__ == "__main__":
    main()
