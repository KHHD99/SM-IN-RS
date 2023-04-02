# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


logger = logging.getLogger()


def exponential_decay(value, max_val, half_life):
    """Compute decay factor for a given value based on an exponential decay.

    Values greater than `max_val` will be set to 1.

    Args:
        value (numeric): Value to calculate decay factor
        max_val (numeric): Value at which decay factor will be 1
        half_life (numeric): Value at which decay factor will be 0.5

    Returns:
        float: Decay factor
    """
    return np.minimum(1.0, np.power(0.5, (max_val - value) / half_life))


def jaccard(cooccurrence):
    """Helper method to calculate the Jaccard similarity of a matrix of co-occurrences.
    When comparing Jaccard with count co-occurrence and lift similarity, count favours
    predictability, meaning that the most popular items will be recommended most of
    the time. Lift, by contrast, favours discoverability/serendipity, meaning that an
    item that is less popular overall but highly favoured by a small subset of users
    is more likely to be recommended. Jaccard is a compromise between the two.

    Args:
        cooccurrence (numpy.ndarray): the symmetric matrix of co-occurrences of items.

    Returns:
        numpy.ndarray: The matrix of Jaccard similarities between any two items.
    """

    diag = cooccurrence.diagonal()

    diag_rows = np.expand_dims(diag, axis=0)

    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cooccurrence / (diag_rows + diag_cols - cooccurrence)
    # print(result)

    return np.array(result)


def lift(cooccurrence):
    """Helper method to calculate the Lift of a matrix of co-occurrences. In comparison
    with basic co-occurrence and Jaccard similarity, lift favours discoverability and
    serendipity, as opposed to co-occurrence that favours the most popular items, and
    Jaccard that is a compromise between the two.

    Args:
        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.

    Returns:
        numpy.ndarray: The matrix of Lifts between any two items.
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cooccurrence / (diag_rows * diag_cols)
    # print(result)

    return np.array(result)


def get_top_k_scored_items(scores, top_k, sort_top_k=False):
    """Extract top K items from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (numpy.ndarray): Score matrix (users x items).
        top_k (int): Number of top items to recommend.
        sort_top_k (bool): Flag to sort top k results.

    Returns:
        numpy.ndarray, numpy.ndarray:
        - Indices into score matrix for each users top items.
        - Scores corresponding to top items.

    """

    # ensure we're working with a dense ndarray
    if isinstance(scores, sparse.spmatrix):
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            "Number of items is less than top_k, limiting top_k to number of items"
        )
    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]

    return np.array(top_items), np.array(top_scores)


def binarize(a, threshold):
    """Binarize the values.

    Args:
        a (numpy.ndarray): Input array that needs to be binarized.
        threshold (float): Threshold below which all values are set to 0, else 1.

    Returns:
        numpy.ndarray: Binarized array.
    """
    return np.where(a > threshold, 1.0, 0.0)


def rescale(data, new_min=0, new_max=1, data_min=None, data_max=None):
    """Rescale/normalize the data to be within the range `[new_min, new_max]`
    If data_min and data_max are explicitly provided, they will be used
    as the old min/max values instead of taken from the data.

    .. note::
        This is same as the `scipy.MinMaxScaler` with the exception that we can override
        the min/max of the old scale.

    Args:
        data (numpy.ndarray): 1d scores vector or 2d score matrix (users x items).
        new_min (int|float): The minimum of the newly scaled data.
        new_max (int|float): The maximum of the newly scaled data.
        data_min (None|number): The minimum of the passed data [if omitted it will be inferred].
        data_max (None|number): The maximum of the passed data [if omitted it will be inferred].

    Returns:
        numpy.ndarray: The newly scaled/normalized data.
    """
    data_min = data.min() if data_min is None else data_min
    data_max = data.max() if data_max is None else data_max
    return (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min


# ------------------------------------------------------ here we added the similarety metrics------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------


# metrics without D
def HD_Jaccard(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / (a + b + c)
    # print(result)
    return np.array(result)
    
def Dice(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (2 * a) / (2 * a + b + c)
    return np.array(result)

def jaccard_3w(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (3 * a) / (3 * a + b + c)
    return np.array(result)

def sokal_sneath_i(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / (a + 2 * b + 2 * c)
    return np.array(result)

def cosine(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / (np.sqrt((a + b) * (a + c)))  
    return np.array(result)
 
def sorgenfrei(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a ** 2) / ((a + b) * (a + c))
    return np.array(result)

def mountford(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / (0.5 * (a * b + a * c) + b * c)
    return np.array(result)

def mcconnaughey(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = ((a ** 2) - b * c) / ((a + b) * (a + c))
    return np.array(result)

def kulczynski_ii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = ((a / 2) * (2 * a + b + c)) / ((a + b) * (a + c))
    return np.array(result)

def driver_kroeber(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a / 2) * (1 / (a + b) + 1 / (a + c))
    return np.array(result)

def johnson(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a / (a + b) + a / (a + c))
    return np.array(result)

def simpson(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    ab = (a + b)
    ac = (a + c)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / np.minimum(ab, ac)
    return np.array(result)

def braun_banquet(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    ab = (a + b)
    ac = (a + c)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / np.maximum(ab, ac)
    return np.array(result)

def fager_mcgowan(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    ab = (a + b)
    ac = (a + c)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a / np.sqrt((a + b) * (a + c))) - np.maximum(ab, ac) / 2
    return np.array(result)

# Distance Without D

def euclid(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.sqrt(b + c)
    scaler = StandardScaler()
    scaler.fit(result)
    result = scaler.transform(result)
    return np.array(1 / (1 + result))

def minkowski(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (b + c)  # =(b+c)**(1/1)
    scaler = StandardScaler()
    scaler.fit(result)
    result = scaler.transform(result)
    return np.array(1 / (1 + result))

def lance_williams(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (b + c) / (2 * a + b + c)
    return np.array(1 - result)

def hellinger(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 2 * np.sqrt(1 - a / np.sqrt((a + b) * (a + c)))
    return np.array(1 / (1 + result))

def chord(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.sqrt(2 * (1 - a / np.sqrt((a + b) * (a + c))))
    return np.array(1 / (1 + result))


######################################################## SIMILARITIES   WITH D ################################################################

# metrics with D
def sokal_michener(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a + d) / (a + b + c + d)
    return np.array(result)

def sokal_sneath_ii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = 2 * (a + d) / (2 * (a + d) + b + c)
    return np.array(result)

def sokal_sneath_iv(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a / (a + b) + a / (a + c) + d / (d + b) + d / (d + c)) / 4
    return np.array(result)

def sokal_sneath_v(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * d) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
    return np.array(result)

def pearson_i(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        result = n * ((a * d - b * c) ** 2) / ((a + b) * (a + c) * (d + b) * (d + c))
    return np.array(result)

def pearson_ii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        alfa = n * ((a * d - b * c) ** 2) / ((a + b) * (a + c) * (d + b) * (d + c))
        result = alfa / (n + alfa)
    return np.sqrt(result)

def pearson_iii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
        result = (rho / (n + rho)) ** 2
    return np.array(result)

def pearson_heron_i(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
    return np.array(result)

def pearson_heron_ii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.cos(np.pi * np.sqrt(b * c) / np.sqrt(a * d) + np.sqrt(b * c))
    return np.array(result)

def baroni_urbani_buser_i(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (np.sqrt(a * d) + a) / (np.sqrt(a * d) + a + b + c)
    return np.array(result)

def baroni_urbani_buser_ii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (np.sqrt(a * d) + a - (b + c)) / (np.sqrt(a * d) + a + b + c)
    return np.array(result)

def forbes_i(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (n * a) / ((a + b) * (a + c))
    # print(result)
    return np.array(result)

def forbes_ii(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d
    ab = (a + b)
    ac = (a + c)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (n * a - (a + b) * (a + c)) / (n * np.minimum(ab, ac) - (a + b) * (a + c))
    return np.array(result)

def yuleq(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * d - b * c) / (a * d + b * c)
    return np.array(result)

def yuleq_w(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (np.sqrt(a * d) - np.sqrt(b * c)) / (np.sqrt(a * d) + np.sqrt(b * c))
    return np.array(result)

def tarantula(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * (c + d)) / (c * (a + b))
    return np.array(result)

def ample(cooccurrence):
    return abs(tarantula(cooccurrence))

def rogers_tanimoto(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a + d) / (a + 2 * (b + c) + d)
    # print(result)
    return np.array(result)

def faith(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a + 0.5 * d) / (a + b + c + d)
    # print(result)
    return np.array(result)

def gower_legendre(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a + d) / (a + 0.5 * (b + c) + d)
    return np.array(result)

def innerproduct(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = a + d
    return np.array(result)

def russell_rao(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = a / (a + b + c + d)
    return np.array(result)

def tarwid(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (n * a - (a + b) * (a + c)) / (n * a + (a + b) * (a + c))
    return np.array(result)

def dennis(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * d - b * c) / (np.sqrt(n * (a + b) * (a + c)))
    return np.array(result)

def gower(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a + d) / np.sqrt((a + b) * (a + c) * (d + b) * (d + c))
    return np.array(result)

def stiles(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.log((n * (abs(a * d - b * c) - n / 2) ** 2) / ((a + b) * (a + c) * (d + b) * (d + c)))
    return np.array(result)

def fossum(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (n * (a - np.full(shape=a.shape, fill_value=0.5)) ** 2) / ((a + b) * (a + c))
    return np.array(result)

def disperson(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * d - b * c) / (a + b + c + d) ** 2
    return np.array(result)

def hamann(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = ((a + d) - (b + c)) / (a + b + c + d)
    return np.array(result)

def michael(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    with np.errstate(invalid="ignore", divide="ignore"):
        result = 4 * (a * d - b * c) / ((a + d) ** 2 + (b + c) ** 2)
    return np.array(result)

def peirce(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (a * b + b * c) / (a * b + 2 * b * c + c * d)
    return np.array(result)

def eyraud(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (n ** 2) * (n * a - (a + b) * (a + c)) / ((a + b) * (a + c) * (d + b) * (d + c))
    return np.array(result)


################################ S=1-D or 1/D ##############################
# Distance With D


def yuleq_d(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (2 * b * c) / (a * d + b * c)
    return np.array(1 - result)

def mean_manhattan(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (b + c) / (a + b + c + d)
    return np.array(1 - result)

def vari(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (b + c) / 4 * (a + b + c + d)
    scaler = MinMaxScaler()
    scaler.fit(result)
    result = scaler.transform(result)
    return np.array(1 / (1 + result))

def shapedifference(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    n = a + b + c + d

    with np.errstate(invalid="ignore", divide="ignore"):
        result = (n * (b + c) - ((b - c) ** 2)) / (a + b + c + d) ** 2
    return np.array(1 - result)

def patterndifference(cooccurrence):
    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)
    a = cooccurrence
    b = diag_rows - cooccurrence
    c = diag_cols - cooccurrence
    d = cooccurrence.shape[0] * np.ones((cooccurrence.shape[0], cooccurrence.shape[0])) - a - b - c
    with np.errstate(invalid="ignore", divide="ignore"):
        result = (4 * b * c) / (a + b + c + d) ** 2
    return np.array(1 - result)
    
   