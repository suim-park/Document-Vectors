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
    pre_trained_model = api.load("word2vec-google-news-300")
    # pre_trained_model = api.load("fasttext-wiki-news-subwords-300")  # Different pre-trained model

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
raw counts (train): 0.9895557029177718 -> 98.956 %
raw counts (test): 0.9582089552238806  -> 95.821 %
---------------------------------------------------
tfidf (train): 0.9993368700265252      -> 99.934 %
tfidf (test): 0.9716417910447761       -> 97.164 %
---------------------------------------------------
lsa (train): 0.9661803713527851        -> 96.618 %
lsa (test): 0.9611940298507463         -> 96.119 %
---------------------------------------------------
word2vec (train): 0.935842175066313    -> 93.584 %
word2vec (test): 0.9298507462686567    -> 92.985 %
```

`Result 2`
```
raw counts (train): 0.9892241379310345 -> 98.922 %
raw counts (test): 0.9671641791044776  -> 96.716 %
---------------------------------------------------
tfidf (train): 0.9993368700265252      -> 99.934 %
tfidf (test): 0.9686567164179104       -> 96.866 %
---------------------------------------------------
lsa (train): 0.9678381962864722        -> 96.784 %
lsa (test): 0.9507462686567164         -> 95.075 %
---------------------------------------------------
word2vec (train): 0.9370026525198939   -> 93.700 %
word2vec (test): 0.9119402985074627    -> 91.194 %
```

`Result 3`
```
raw counts (train): 0.9885610079575596 -> 98.856 %
raw counts (test): 0.9567164179104478  -> 95.672 %
---------------------------------------------------
tfidf (train): 0.9991710875331565      -> 99.917 %
tfidf (test): 0.9626865671641791       -> 96.269 %
---------------------------------------------------
lsa (train): 0.9681697612732095        -> 96.817 %
lsa (test): 0.9537313432835821         -> 95.373 %
---------------------------------------------------
word2vec (train): 0.9356763925729443   -> 93.568 %
word2vec (test): 0.9208955223880597    -> 92.090 %
```

__`Case 2`__
* Word2Vec pre-trained model: _fasttext-wiki-news-subwords-300_ </br>

`Result 1`
```
raw counts (train): 0.9890583554376657 -> 98.906 %
raw counts (test): 0.9731343283582089  -> 97.313 %
---------------------------------------------------
tfidf (train): 0.9993368700265252      -> 99.934 %
tfidf (test): 0.9611940298507463       -> 96.120 %
---------------------------------------------------
lsa (train): 0.9658488063660478        -> 96.585 %
lsa (test): 0.9597014925373134         -> 95.970 %
---------------------------------------------------
word2vec (train): 0.9673408488063661   -> 96.734 %
word2vec (test): 0.9492537313432836    -> 94.925 %
```

`Result 2`
```
raw counts (train): 0.9897214854111406 -> 98.972 %
raw counts (test): 0.9701492537313433  -> 97.015 %
---------------------------------------------------
tfidf (train): 0.9993368700265252      -> 99.934 %
tfidf (test): 0.9656716417910448       -> 96.567 %
---------------------------------------------------
lsa (train): 0.9673408488063661        -> 96.734 %
lsa (test): 0.9567164179104478         -> 95.672 %
---------------------------------------------------
word2vec (train): 0.9661803713527851   -> 96.618 %
word2vec (test): 0.9537313432835821    -> 95.373 %
```

`Result 3`
```
raw counts (train): 0.9897214854111406 -> 98.972 %
raw counts (test): 0.9597014925373134  -> 95.970 %
---------------------------------------------------
tfidf (train): 0.9993368700265252      -> 99.934 %
tfidf (test): 0.9686567164179104       -> 96.866 %
---------------------------------------------------
lsa (train): 0.9694960212201591        -> 96.950 %
lsa (test): 0.9373134328358209         -> 93.731 %
---------------------------------------------------
word2vec (train): 0.9661803713527851   -> 96.618 %
word2vec (test): 0.9477611940298507    -> 94.776 %
```

## Explanation of Results
### Hypothesis
  ```
  Function Performance: word2vec -> LSA -> tf-idf -> raw counts
  ```
### Analysis of Results
By comparing the results of four different methods on document vectors, we observed several outcomes. Initially, we hypothesized that the performance would degrade in the order of word2vec, LSA, tf-idf, and raw counts. However, the actual results, derived from comparing the percent correct using training datasets from Austen's "Sense and Sensibility", Carroll's "Alice in Wonderland", "Google News" and "Fasttext Wiki News", contradicted our expectations. There are several reasons that might explain this discrepancy.</br>
First of all, unlike LSA, tf-idf, and raw counts, the word2vec model was trained not using combined data from Austen and Carroll, but using "Google News" or "Fasttext Wiki News". Hence, the correct percentage appeared lower than anticipated. Despite the difference in training models, a result of about 92-95% demonstrates the excellent correct performance of word2vec. Additionally, the performance of word2vec varied based on the document we used. Utilizing "fasttext wiki new subwords" as a pre-trained model resulted in a 2-3% higher performance compared to when we used "Google News". Another contributing factor is the higher accuracy rate of tf-idf and raw counts compared to word2vec or LSA, which can arise due to various complex reasons. One primary reason could be the dataset size. Word2vec requires large data, whereas raw counts, being a simple technique, can work well even with smaller datasets. The dataset size of Google News used for word2vec is around 1.6GB, and that of fasttext wiki news is about 1GB; significantly smaller than the typical dataset sizes for word2vec. Hence, this size difference could have influenced the results.</br>
Moreover, the nature of the documents can impact the results. TF-IDF emphasizes words frequently appearing in specific documents, and raw counts consider only the frequency of word appearance. If the training dataset has documents with such features prominently, then they can display better accuracy than word2vec and LSA. Furthermore, both word2vec and LSA require model tuning to find the optimal settings, which might make simpler models like td-idf and raw counts perform better in certain cases.</br>
Therefore, we can see that the actual results differ from the initially hypothesized performance order, and we can understand the reasons behind this discrepancy.

### Summary
By comparing the percentage correct through four different word/document embeddings methods, it's evident that methods which are theoretically expected to show a higher match rate, based on various factors like training data documents, data size, document characteristics, model configurations, and etc., might sometimes exhibit a lower match rate than relatively simple methods. Therefore, if one configures the model considering all these factors, a higher match rate can be achieved.
