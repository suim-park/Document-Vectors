"""Compare token/document vectors for classification."""
import random
from typing import List, Mapping, Optional, Sequence

import gensim
import gensim.downloader as api
from gensim.models import Word2Vec
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

FloatArray = NDArray[np.float64]

# Un-comment this to fix the random seed
# random.seed(31)

austen = nltk.corpus.gutenberg.sents("austen-sense.txt")
carroll = nltk.corpus.gutenberg.sents("carroll-alice.txt")

vocabulary = sorted(
    set(token for sentence in austen + carroll for token in sentence)
) + [None]
vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}


def onehot(
    vocabulary_map: Mapping[Optional[str], int], token: Optional[str]
) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map),))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx] = 1
    return embedding


def sum_token_embeddings(
    token_embeddings: Sequence[FloatArray],
) -> FloatArray:
    """Sum the token embeddings."""
    total: FloatArray = np.array(token_embeddings).sum(axis=0)
    return total


def split_train_test(
    X: FloatArray, y: FloatArray, test_percent: float = 10
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Split data into training and testing sets."""
    N = len(y)
    data_idx = list(range(N))
    random.shuffle(data_idx)
    break_idx = round(test_percent / 100 * N)
    training_idx = data_idx[break_idx:]
    testing_idx = data_idx[:break_idx]
    X_train = X[training_idx, :]
    y_train = y[training_idx]
    X_test = X[testing_idx, :]
    y_test = y[testing_idx]
    return X_train, y_train, X_test, y_test


def generate_data_token_counts(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with raw token counts."""
    X: FloatArray = np.array(
        [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in h0_documents
        ]
        + [
            sum_token_embeddings([onehot(vocabulary_map, token) for token in sentence])
            for sentence in h1_documents
        ]
    )
    y: FloatArray = np.array(
        [0 for sentence in h0_documents] + [1 for sentence in h1_documents]
    )
    return split_train_test(X, y)


def generate_data_tfidf(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with TF-IDF scaling."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        h0_documents, h1_documents
    )
    tfidf = TfidfTransformer(norm=None).fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)
    return X_train, y_train, X_test, y_test


def generate_data_lsa(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with LSA."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        h0_documents, h1_documents
    )
    #########################################################################
    lsa = TruncatedSVD(
        n_components=300
    )  # Setting the length of the document vector to 300
    X_train = lsa.fit_transform(X_train)

    # Applying LSA to the test data using the trained LSA transformer
    X_test = lsa.transform(X_test)
    #########################################################################
    return X_train, y_train, X_test, y_test


def generate_data_word2vec(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with word2vec."""
    #########################################################################
    # Load the pre-trained word2vec model
    # It can be substituted with another pre-trained model if desired

    pre_trained_model = api.load("word2vec-google-news-300")
    # pre_trained_model = api.load("fasttext-wiki-news-subwords-300")  # Different pre-trained model

    # Function to create document embeddings
    def document_embedding(doc: list[str], model) -> FloatArray:
        embeddings = [model[word] for word in doc if word in model]
        if not embeddings:
            return np.zeros(model.vector_size)
        return np.sum(embeddings, axis=0)

    # Generate embeddings for each document
    X_h0 = np.array(
        [document_embedding(doc, pre_trained_model) for doc in h0_documents]
    )
    X_h1 = np.array(
        [document_embedding(doc, pre_trained_model) for doc in h1_documents]
    )

    # Merge data arrays
    X = np.vstack((X_h0, X_h1))
    y = np.array([0] * len(h0_documents) + [1] * len(h1_documents))
    #########################################################################
    return split_train_test(X, y)


def run_experiment() -> None:
    """Compare performance with different embeddiings."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("raw counts (train):", clf.score(X_train, y_train))
    print("raw counts (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_tfidf(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("tfidf (train):", clf.score(X_train, y_train))
    print("tfidf (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_lsa(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("lsa (train):", clf.score(X_train, y_train))
    print("lsa (test):", clf.score(X_test, y_test))

    X_train, y_train, X_test, y_test = generate_data_word2vec(austen, carroll)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
    print("word2vec (train):", clf.score(X_train, y_train))
    print("word2vec (test):", clf.score(X_test, y_test))


if __name__ == "__main__":
    run_experiment()
