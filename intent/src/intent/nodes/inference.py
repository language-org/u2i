import os

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance

from intent.src.intent.nodes import similarity


def label_queries(
    text: tuple, DIST_THRES: float, verbose: bool = False
) -> pd.DataFrame:
    """Label text queries using semantic similarity-based hierarchical clustering

    Args:
        text (tuple): tuples of text queries
        DIST_THRES (float):
            hierarchical clustering thresholding parameter
            distance threshold t - the maximum inter-cluster distance allowed
        verbose (bool, optional): print or not. Defaults to False.

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
    sim_mtx = similarity.get_semantic_similarity_matrix(text)
    sim_mtx = pd.DataFrame(sim_mtx)

    # patch weird values with -1
    sim_mtx[np.logical_or(sim_mtx < 0, sim_mtx > 1)] = -0.1
    sim_mtx[sim_mtx.isnull()] = -1
    row_linkage = hierarchy.linkage(distance.pdist(sim_mtx), method="average")
    if verbose:
        sns.clustermap(
            sim_mtx,
            row_linkage=row_linkage,
            method="average",
            figsize=(13, 13),
            cmap="vlag",
        )
    label = fcluster(row_linkage, t=DIST_THRES, criterion="distance")
    if verbose:
        print(f"{round(time() - t0, 2)} secs")

    return pd.DataFrame([text, label]).T.rename(columns={0: "query", 1: "label"})
