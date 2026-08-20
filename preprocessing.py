# ==========================================
# PREPROCESSING.PY
# SKRIPSI MBG
# ==========================================

import re
import string
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# download nltk (aman kalau sudah ada)
nltk.download("punkt")
nltk.download("stopwords")


# ==========================================
# STEMMER
# ==========================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()


# ==========================================
# STOPWORD INDONESIA
# (sesuaikan notebook terbaru)
# ==========================================

stopword_indonesia = stopwords.words("indonesian")


custom_stopword = [

    # kata umum
    "yang", "dengan", "dari", "untuk",
    "dalam", "oleh", "ke", "di",

    # twitter artifacts
    "link", "url", "https", "http",
    "co", "amp", "rt",

    # filler words
    "nih", "dong", "deh", "lah",
    "aja", "kok", "yah", "ya",
    "sih", "pun", "nya",

    # mbg frequent noise
    "program", "gratis"

]


all_stopwords = list(
    set(stopword_indonesia + custom_stopword)
)


# ==========================================
# NORMALIZATION DICTIONARY
# (ambil dari notebook kalau ada tambahan)
# ==========================================

normalization_dict = {

    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "bgt": "banget",
    "yg": "yang",
    "dr": "dari",
    "utk": "untuk",
    "tp": "tapi",
    "krn": "karena",
    "dgn": "dengan",

    "bgus": "bagus",
    "bantuin": "membantu",

    "anak2": "anak",
    "org": "orang"

}


# ==========================================
# STEP 1 CLEANING
# ==========================================

def cleaning_text(text):

    text = str(text)

    # hapus url
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # hapus mention
    text = re.sub(
        r"@\w+",
        "",
        text
    )

    # hapus hashtag symbol
    text = re.sub(
        r"#",
        "",
        text
    )

    # hapus angka
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # hapus emoji/unicode
    text = re.sub(
        r"[^\x00-\x7F]+",
        "",
        text
    )

    # hapus punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # hapus whitespace berlebih
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================================
# STEP 2 CASE FOLDING
# ==========================================

def case_folding(text):

    return text.lower()


# ==========================================
# STEP 3 TOKENIZING
# ==========================================

def tokenizing(text):

    return word_tokenize(text)


# ==========================================
# STEP 4 NORMALIZATION
# ==========================================

def normalization(tokens):

    normalized = []

    for word in tokens:

        if word in normalization_dict:
            normalized.append(
                normalization_dict[word]
            )
        else:
            normalized.append(word)

    return normalized


# ==========================================
# STEP 5 STOPWORD REMOVAL
# ==========================================

def stopword_removal(tokens):

    filtered = [

        word for word in tokens

        if word not in all_stopwords
    ]

    return filtered


# ==========================================
# STEP 6 STEMMING
# ==========================================

def stemming(tokens):

    stemmed = [

        stemmer.stem(word)

        for word in tokens
    ]

    return stemmed


# ==========================================
# PIPELINE PREPROCESSING
# FINAL OUTPUT
# ==========================================

def preprocess_text(text):

    clean = cleaning_text(text)

    folded = case_folding(clean)

    tokens = tokenizing(folded)

    normalized = normalization(tokens)

    filtered = stopword_removal(normalized)

    stemmed = stemming(filtered)

    final_text = " ".join(stemmed)

    return final_text


# ==========================================
# PREPROCESSING DEMO
# untuk streamlit page
# menampilkan step by step
# ==========================================

def preprocess_with_steps(text):

    original = str(text)

    clean = cleaning_text(original)

    folded = case_folding(clean)

    tokens = tokenizing(folded)

    normalized = normalization(tokens)

    filtered = stopword_removal(normalized)

    stemmed = stemming(filtered)

    final_text = " ".join(stemmed)

    result = {

        "original": original,

        "cleaning": clean,

        "case_folding": folded,

        "tokenizing": tokens,

        "normalization": normalized,

        "stopword_removal": filtered,

        "stemming": stemmed,

        "final_text": final_text

    }

    return result


# ==========================================
# BULK PREPROCESS DATAFRAME
# ==========================================

def preprocess_dataframe(df, text_column):

    df = df.copy()

    df["hasil_preprocessing"] = df[
        text_column
    ].apply(
        preprocess_text
    )

    return df


# ==========================================
# TEST LOCAL
# ==========================================

if __name__ == "__main__":

    sample = "Program MBG sangat bagus dan membantu siswa!!!"

    result = preprocess_with_steps(sample)

    for key, value in result.items():

        print(key, ":", value)