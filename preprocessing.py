import os
import re
import unicodedata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

RANDOM_SEED = 42

REMOVE_NAMES = True  #switch to remove hotels and city names so we can check the impact of this change
MIN_DF = 5   # sparse-term removal: keep only words that appear in at least 5 training reviews
FIG_DIR = "figures"   # plots are saved here instead of shown
os.makedirs(FIG_DIR, exist_ok=True)

pd.set_option("display.max_colwidth", 200)


def load_data(path="data/all_data.csv"):
    df = pd.read_csv(path, index_col=0)   # first column is just a row index
    df = df[df["Review_Language"] == "English"].reset_index(drop=True)

    print(df.shape)
    print(pd.crosstab(df["source"], df["Sentiment"], margins=True))
    print("\nMissing values per column:")
    print(df.isna().sum())
    return df


def missing_text_per_class(df):
    missing = (
        df.assign(
            up_missing=df["Upside_Review"].isna(),
            down_missing=df["Downside_Review"].isna(),
            both_missing=df["Upside_Review"].isna() & df["Downside_Review"].isna(),
        )
        .groupby(["source", "Sentiment"])[["up_missing", "down_missing", "both_missing"]]
        .mean()
        .round(3)
    )
    print(missing)

    # The na_* flags in the csv do not perfectly match the NaNs.  We use the NaNs themselves
    print("\nMismatch na_up_review vs NaN:  ", (df["Upside_Review"].isna() != df["na_up_review"]).sum())
    print("Mismatch na_down_review vs NaN:", (df["Downside_Review"].isna() != df["na_down_review"]).sum())


def most_frequent_texts(df):
    for col in ["Upside_Review", "Downside_Review"]:
        print(f"\n{col} - most frequent texts per class")
        for s, name in [(0, "Human"), (1, "AI")]:
            top = df.loc[df["source"] == s, col].str.strip().str.lower().value_counts().head(5)
            print(f"  {name}: {top.to_dict()}")


def build_text_raw(df):
    df["text_raw"] = (
        df["Upside_Review"].fillna("").str.strip() + " " + df["Downside_Review"].fillna("").str.strip()
    ).str.strip()

    print("Empty reviews:     ", (df["text_raw"] == "").sum())
    print("Duplicate reviews: ", df["text_raw"].duplicated().sum())
    return df


def review_length(df):
    df["n_words"] = df["text_raw"].str.split().str.len()
    print(df.groupby(["source", "Sentiment"])["n_words"].describe().round(1))

    fig, ax = plt.subplots(figsize=(7, 4))
    for s, name in [(0, "Human"), (1, "AI")]:
        ax.hist(df.loc[df["source"] == s, "n_words"], bins=np.arange(0, 300, 10), alpha=0.6, label=name)
    ax.set_xlabel("Number of words per review")
    ax.set_ylabel("Count")
    ax.set_title("Review length by class")
    ax.legend()
    plt.savefig(f"{FIG_DIR}/review_length.png", dpi=200, bbox_inches="tight")
    plt.close()
    return df


def surface_cues(df):
    def mentions(col):
        return pd.Series([str(n).lower() in t.lower() for n, t in zip(df[col], df["text_raw"])])

    checks = pd.DataFrame({
        "has_digit":           df["text_raw"].str.contains(r"\d"),
        "has_newline":         df["text_raw"].str.contains("\n"),
        "curly_apostrophe":    df["text_raw"].str.contains("’"),
        "straight_apostrophe": df["text_raw"].str.contains("'"),
        "non_ascii":           df["text_raw"].str.contains(r"[^\x00-\x7f]"),
        "mentions_hotel":      mentions("Hotel Name"),
        "mentions_city":       mentions("City Name"),
    })
    print(checks.groupby(df["source"]).mean().T.rename(columns={0: "Human", 1: "AI"}).round(3))


def preprocess(text, hotel=None, city=None):
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")   #curly to straight apostrophe
    if REMOVE_NAMES:
        for name in (hotel, city):
            if isinstance(name, str) and name:
                text = text.replace(name.lower(), " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()   #strip accents
    text = re.sub(r"\d+(?:[.,]\d+)*", " num ", text)      # numbers -> 'num'
    text = re.sub(r"[^a-z' ]+", " ", text)             # keep leters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(df):
    df["text"] = [preprocess(t, h, c) for t, h, c in zip(df["text_raw"], df["Hotel Name"], df["City Name"])]

    print("Empty after cleaning:", (df["text"] == "").sum())
    print(df.sample(5, random_state=RANDOM_SEED)[["source", "Sentiment", "text_raw", "text"]])
    return df


def get_model_inputs(df):
    X_text    = df["text"]
    y         = df["source"].astype(int)     # 1=AI , 0=human
    sentiment = df["Sentiment"]              # needed for q3

    print(X_text.shape, y.value_counts().to_dict())
    return X_text, y, sentiment


def split_data(X_text, y, sentiment):
    X_train, X_test, y_train, y_test, sentiment_train, sentiment_test = train_test_split(X_text, y, sentiment, test_size = 0.25, random_state = RANDOM_SEED, stratify = y)

    print(y_test.value_counts().to_dict())
    return X_train, X_test, y_train, y_test, sentiment_train, sentiment_test


def bag_of_words(X_train, X_test):
    vectorizer = CountVectorizer(min_df=MIN_DF)
    vectorizer.fit(X_train)   # vocabulary from the training set only; transform gives both matrices the same columns
    train_bow = vectorizer.transform(X_train)
    test_bow = vectorizer.transform(X_test)
    print(train_bow.shape, test_bow.shape)
    return vectorizer, train_bow, test_bow
