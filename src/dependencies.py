"""Helper functions."""

# Author: Andreas Gompos <andreas.gompos@gmail.com>

import os

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer


def load_glove_embeddings(glove_embedding_dir):
    print("loading glove embeddings started")
    glove_embeddings = {}
    f = open(glove_embedding_dir, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        glove_embeddings[word] = coefs
    f.close()
    print("loading glove embeddings finished")
    return glove_embeddings


def load_dataset(directory):
    df = pd.DataFrame()
    summ = 0
    classes = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
    ]

    print("Loading:")
    for class_ in classes:
        current_class_directory = "{}{}/".format(directory, class_)
        print(current_class_directory)

        for name in sorted(os.listdir(current_class_directory)):
            path = os.path.join(current_class_directory, name)

            current_text = open(path, encoding="ISO-8859-1")
            summ += 1
            df.loc[summ, "text"] = current_text.read()
            df.loc[summ, "class"] = class_

    df["class_meaning"] = df["class"]
    df["class"].replace(
        {
            "business": 0,
            "entertainment": 1,
            "politics": 2,
            "sport": 3,
            "tech": 4
        },
        inplace=True,
    )
    df = df.sample(frac=1, random_state=25)
    return df


def tokenize_document(document):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokenizer_ = RegexpTokenizer("[a-zA-Z]+")

    words = []
    for sentence in sent_tokenize(document):
        tokens = [
            lemmatizer.lemmatize(t.lower())
            for t in tokenizer_.tokenize(sentence)
            if t.lower() not in stop_words
        ]
        words += tokens

    words_ = str()
    for word in words:
        words_ = words_ + " " + word
    return words_


class DocTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        X = pd.Series(X)
        return X.apply(tokenize_document).tolist()


class WordsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, top_words=20000):
        self.top_words = top_words

    def fit(self, X, y=None, **fit_params):
        encoder = Tokenizer(self.top_words)
        encoder.fit_on_texts(X)
        self.encoder_ = encoder

        return self

    def transform(self, X, **transform_params):

        return self.encoder_.texts_to_sequences(X)


class Padder(BaseEstimator, TransformerMixin):
    def __init__(self, max_sequence_length=500):
        self.max_sequence_length = max_sequence_length

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **transform_params):
        return sequence.pad_sequences(
            np.array(X), maxlen=self.max_sequence_length)
