# author: steeve laquitaine

import os
from time import time

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

from intent.src.intent.nodes import similarity


def cluster_queries(
    text: pd.Series, dist_thresh: float, verbose: bool = False, hcl_method=None
) -> pd.DataFrame:
    """Label text queries using semantic similarity-based hierarchical clustering

    Args:
        text (pd.Series): series of text queries with their raw indices
        DIST_THRES (float):
            hierarchical clustering thresholding parameter
            distance threshold t - the maximum inter-cluster distance allowed
        verbose (bool, optional): print or not. Defaults to False.
        hcl_method (str):

    Returns:
        pd.DataFrame: dataframe of queries with their cluster label

    Usage:
        text = (
        "want to drink a cup of coffee",
        "want to drink a cup of tea",
        "would like a bottle of water",
        "want to track my credit card"
        )
        df = label_queries(text, 1.8)
    """
    t0 = time()

    # convert pandas series to tuple
    text_tuple = tuple(text)

    # compute query similarity matrix
    sim_mtx = similarity.get_semantic_similarity_matrix(text_tuple)
    sim_mtx = pd.DataFrame(sim_mtx)

    # patch weird values with -1
    sim_mtx[np.logical_or(sim_mtx < 0, sim_mtx > 1)] = -0.1
    sim_mtx[sim_mtx.isnull()] = -1

    # apply hierarchical clustering to matrix
    row_linkage = hierarchy.linkage(distance.pdist(sim_mtx), method=hcl_method)
    if verbose:
        sns.clustermap(
            sim_mtx,
            row_linkage=row_linkage,
            method="average",
            figsize=(13, 13),
            cmap="vlag",
        )
    label = fcluster(row_linkage, t=dist_thresh, criterion="distance")
    if verbose:
        print(f"{round(time() - t0, 2)} secs")

    # convert to dataframe
    labelled = pd.DataFrame([text_tuple, label]).T.rename(
        columns={0: "query", 1: "label"}
    )

    # keep corpus indices
    labelled.index = text.index

    # sort by label
    labelled_sorted = labelled.sort_values(by=["label"])

    return labelled_sorted
