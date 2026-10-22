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

pd.set_option("display.max_colwidth", 200)


def load_data(path="data/all_data.csv"):
    """Load the reviews and keep only the English ones.

    Columns in 'all_data.csv' that matter for us:
    - 'Review_Language': keep only 'English'
    - 'Upside_Review', 'Downside_Review': the review text is split in a positive and a negative part (either can be empty/NaN)
    - 'Sentiment': either 'POS' or 'NEG' (question 3)
    - 'source': the class label (1 = AI, 0 = Human)
    - 'Prompt_Language': only filled for AI reviews
    """
    df = pd.read_csv(path, index_col=0)   # first column is just a row index
    df = df[df["Review_Language"] == "English"].reset_index(drop=True)

    print(df.shape)
    print(pd.crosstab(df["source"], df["Sentiment"], margins=True))
    print("\nMissing values per column:")
    print(df.isna().sum())
    return df


def missing_text_per_class(df):
    """Missing text per class.

    Only the review text is used as input. 'Review_Score', 'City Name', 'Hotel Name' and 'Prompt_Language'
    are not used as features: the score is not part of the text, and 'Prompt_Language' exists only for
    AI reviews (it would leak the label).
    """
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
    """Most frequent texts per class."""
    for col in ["Upside_Review", "Downside_Review"]:
        print(f"\n{col} - most frequent texts per class")
        for s, name in [(0, "Human"), (1, "AI")]:
            top = df.loc[df["source"] == s, col].str.strip().str.lower().value_counts().head(5)
            print(f"  {name}: {top.to_dict()}")


def build_text_raw(df):
    """Build one text column per review: concatenate upside and downside part. If missing part, use empty string."""
    df["text_raw"] = (
        df["Upside_Review"].fillna("").str.strip() + " " + df["Downside_Review"].fillna("").str.strip()
    ).str.strip()

    print("Empty reviews:     ", (df["text_raw"] == "").sum())
    print("Duplicate reviews: ", df["text_raw"].duplicated().sum())
    return df


def review_length(df):
    """Review length per class / sentiment."""
    df["n_words"] = df["text_raw"].str.split().str.len()
    print(df.groupby(["source", "Sentiment"])["n_words"].describe().round(1))

    fig, ax = plt.subplots(figsize=(7, 4))
    for s, name in [(0, "Human"), (1, "AI")]:
        ax.hist(df.loc[df["source"] == s, "n_words"], bins=np.arange(0, 300, 10), alpha=0.6, label=name)
    ax.set_xlabel("Number of words per review")
    ax.set_ylabel("Count")
    ax.set_title("Review length by class")
    ax.legend()
    plt.show()
    return df


def surface_cues(df):
    """Surface cues that differ between classes."""
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
    """Text cleaning choices:
    - Lowercasing: "Location" and "location" become the same term.
    - Apostrophes normalised.
    - Hotel and city names removed.
    - Accents stripped.
    - Numbers -> 'num' token: we keep this stylistic signal but merge all numbers into one term to avoid many sparse terms.
    - Punctuation, emoji and other non-letter characters removed: the bag-of-words model only uses words.
    - Stop words are kept: function words and pronouns ("i", "we", "the", "very") are known cues for deceptive
      or generated text. Sparse-term removal and feature selection come later and are done on the training set only.
    - No stemming/lemmatization: GPT-4 tends to repeat specific word forms, and keeping them makes the most
      important features easier to read.
    """
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")   #curly to straight apostrophe
    if REMOVE_NAMES:
        for name in (hotel, city):
            if isinstance(name, str) and name:
                text = text.replace(name.lower(), " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()   #strip accents
    text = re.sub(r"\d+(?:[.,]\d+)*", " num ", text)      # numbers -> 'num'
    text = re.sub(r"[^a-z' ]+", " ", text)             # keep letters and apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text(df):
    """Text cleaning."""
    df["text"] = [preprocess(t, h, c) for t, h, c in zip(df["text_raw"], df["Hotel Name"], df["City Name"])]

    print("Empty after cleaning:", (df["text"] == "").sum())
    print(df.sample(5, random_state=RANDOM_SEED)[["source", "Sentiment", "text_raw", "text"]])
    return df


def get_model_inputs(df):
    """Objects used in the rest of the pipeline."""
    X_text    = df["text"]
    y         = df["source"].astype(int)     # 1=AI , 0=human
    sentiment = df["Sentiment"]              # needed for q3

    print(X_text.shape, y.value_counts().to_dict())
    return X_text, y, sentiment


def split_data(X_text, y, sentiment):
    """Stratified 75/25 split.

    1- Split the data randomly into a training/validation set (75%) and a test set (25%).
    2- Use stratified random sampling so that the class distribution in the training set and test set are the same.

    The test set (500 reviews) is only used at the very end, to estimate the performance of the five final models.
    Everything else (vectorizer vocabulary, sparse-term removal, feature selection, hyperparameter tuning) is
    fitted on the training set only.
    """
    X_train, X_test, y_train, y_test, sentiment_train, sentiment_test = train_test_split(X_text, y, sentiment, test_size = 0.25, random_state = RANDOM_SEED, stratify = y)

    print(y_test.value_counts().to_dict())
    return X_train, X_test, y_train, y_test, sentiment_train, sentiment_test


def bag_of_words(X_train, X_test):
    """Represent the review texts using word frequencies (bag-of-words representation)."""
    vectorizer = CountVectorizer(min_df=MIN_DF)
    vectorizer.fit(X_train)   # vocabulary from the training set only; transform gives both matrices the same columns
    train_bow = vectorizer.transform(X_train)
    test_bow = vectorizer.transform(X_test)
    print(train_bow.shape, test_bow.shape)
    return vectorizer, train_bow, test_bow
