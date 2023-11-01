# HW 4. Document Vectors :pencil:
## Information :computer:
* __`Class`__: Introduction to Natural Language Processing
* __`Professor`__: Patrick Wang
* __`Assignment`__: Document Vectors
* __`Name`__: Suim Park(sp699)

## Document Vectors Experiment :pushpin:
Explore how dense word/document embeddings can be used for document classification. You will be
attempting to distinguish between documents from two different authors.
## Question
Use the provided script as a starting point. Before beginning, read and understand what it's doing. Then __implement two types of dense document vectors__:
```
1. using LSA on raw token counts
2. summing pretrained word2vec embeddings
```
Both should produce document vectors of length 300.</br>
__Show and discuss the results.__ The results/discussion should include 
```
1. the percent correct for each method, and
2. a brief explanation of the relative performance, i.e. why might method A lead to better classification performance than method B, etc.
```
You may work in a group of 1 or 2. Submissions will be graded without regard for the group size. You should turn in a document (.txt, .md, or .pdf) answering all of the red items above. You should also turn in a
Python scripts (.py) for the blue items . You may use only the standard library, numpy, sklearn, and gensim.

## Code Generation
__`Packages`__
* The 'gensim.downloader' package has been installed for the word2vec process.
```Python
import random
from typing import List, Mapping, Optional, Sequence

import gensim
import gensim.downloader as api
import nltk
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
```

__`Latent Semantic Analysis(LSA)`__
* The 'generate_data_lsa' function processes two sets of documents ('h0_documents' and 'h1_documents') by converting their token counts into a reduced-dimensional representation using Latent Semantic Analysis (LSA) with a target vector length of 300. It then splits the processed data into training and testing sets for further use in classification tasks.
```Python
def generate_data_lsa(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with LSA."""
    X_train, y_train, X_test, y_test = generate_data_token_counts(
        h0_documents, h1_documents
    )
    #########################################################################
    lsa = TruncatedSVD(n_components=300)  # Setting the length of the document vector to 300
    X_train = lsa.fit_transform(X_train)

    # Applying LSA to the test data using the trained LSA transformer
    X_test = lsa.transform(X_test)
    #########################################################################
    return X_train, y_train, X_test, y_test
```

__`Word2Vec`__ </br>
* The 'generate_data_word2vec' function loads a pre-trained word2vec model and computes document embeddings for two sets of documents ('h0_documents' and 'h1_documents') by averaging the embeddings of the words within them. It then prepares the data for classification, splitting it into training and testing sets using the provided 'split_train_test' function.
```Python
def generate_data_word2vec(
    h0_documents: list[list[str]], h1_documents: list[list[str]]
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Generate training and testing data with word2vec."""
    #########################################################################
    # Load the pre-trained word2vec model
    # It can be substituted with another pre-trained model if desired
    model = api.load("word2vec-google-news-300") # For example

    # Function to create document embeddings
    def document_embedding(doc: list[str], model) -> FloatArray:
        embeddings = [model[word] for word in doc if word in model]
        if not embeddings:
            return np.zeros(model.vector_size)
        return np.sum(embeddings, axis=0)

    # Generate embeddings for each document
    X_h0 = np.array([document_embedding(doc, model) for doc in h0_documents])
    X_h1 = np.array([document_embedding(doc, model) for doc in h1_documents])

    # Merge data arrays
    X = np.vstack((X_h0, X_h1))
    y = np.array([0] * len(h0_documents) + [1] * len(h1_documents))
    #########################################################################
    return split_train_test(X, y)
```

## Results
__`Case 1`__
* Word2Vec pre-trained model: _word2vec-google-news-300_ </br>

`Result 1`
```
raw counts (train): 0.9895557029177718
raw_counts (test): 0.9582089552238806
---------------------------------------
tfidf (train): 0.9993368700265252
tfidf (test): 0.9716417910447761
---------------------------------------
lsa (train): 0.9661803713527851
lsa (test): 0.9611940298507463
---------------------------------------
word2vec (train): 0.935842175066313
word2vec (test): 0.9298507462686567
```

`Result 2`
```
raw counts (train): 0.9892241379310345
raw_counts (test): 0.9671641791044776
---------------------------------------
tfidf (train): 0.9993368700265252
tfidf (test): 0.9686567164179104
---------------------------------------
lsa (train): 0.9678381962864722
lsa (test): 0.9507462686567164
---------------------------------------
word2vec (train): 0.9370026525198939
word2vec (test): 0.9119402985074627
```

`Result 3`
```
raw counts (train): 0.9885610079575596
raw_counts (test): 0.9567164179104478
---------------------------------------
tfidf (train): 0.9991710875331565
tfidf (test): 0.9626865671641791
---------------------------------------
lsa (train): 0.9681697612732095
lsa (test): 0.9537313432835821
---------------------------------------
word2vec (train): 0.9356763925729443
word2vec (test): 0.9208955223880597
```
