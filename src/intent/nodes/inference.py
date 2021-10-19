import os
from collections import Counter
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance
from src.intent.nodes import similarity


def label_queries(
    text: tuple, DIST_THRES: float, verbose: bool = False
) -> pd.DataFrame:
    """Label text queries using semantic similarity-based hierarchical clustering

    [DEPRECATED]

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
    sim_mtx = similarity.get_semantic_similarity_matrix(
        text
    )
    sim_mtx = pd.DataFrame(sim_mtx)

    # patch weird values with -1
    sim_mtx[np.logical_or(sim_mtx < 0, sim_mtx > 1)] = -0.1
    sim_mtx[sim_mtx.isnull()] = -1
    row_linkage = hierarchy.linkage(
        distance.pdist(sim_mtx), method="average"
    )
    if verbose:
        sns.clustermap(
            sim_mtx,
            row_linkage=row_linkage,
            method="average",
            figsize=(13, 13),
            cmap="vlag",
        )
    label = fcluster(
        row_linkage, t=DIST_THRES, criterion="distance"
    )
    if verbose:
        print(f"{round(time() - t0, 2)} secs")

    return pd.DataFrame([text, label]).T.rename(
        columns={0: "query", 1: "cluster_labels"}
    )


class Prediction:
    def __init__(self, method: str):
        self.method = method
        self._print_params()

    def _print_params(self):
        print("(Prediction) method", self.method)

    def _run_cluster_label_mode(
        self, corpus, clustered,
    ):
        """Assign each cluster's most frequent label as predicted class

        label are the true classes

        pros:
            - quick to implement
        cons:
            - wrong in some cases:
                - when there are as many clusters as instances, there will be 100% accuracy
                because each instance will take the true label as its predicted label.

        """

        true_labels = corpus.loc[clustered.index][
            "category"
        ]
        clustered["true_labels"] = true_labels
        clustered

        # label each cluster with its most frequent true label
        unique_labels = clustered["cluster_labels"].unique()
        predicted_labels_all = []
        proba_predicted_all = []
        n_clustered = len(clustered)

        # for each cluster, assign the most frequent true label as prediction
        for ix, this_label in enumerate(unique_labels):

            # find indices of this cluster intents
            this_label_ix = np.where(
                clustered == this_label
            )[0]
            nb_label = len(this_label_ix)

            # find its most frequent true label
            predicted = Counter(
                clustered["true_labels"]
                .iloc[this_label_ix]
                .tolist()
            ).most_common()[0]

            # assign this true label as predicted label and its conditional proba
            # as proba of predicted
            predicted_labels_all += [
                predicted[0]
            ] * nb_label
            proba_predicted_all += [
                predicted[1] / n_clustered
            ] * nb_label
        clustered["predicted"] = predicted_labels_all
        clustered[
            "proba_predicted (ratio)"
        ] = proba_predicted_all
        return clustered

    def run(self, corpus, clustered):

        try:
            if self.method == "cluster_label_mode":
                predictions = self._run_cluster_label_mode(
                    corpus, clustered
                )
        except:
            raise ValueError(
                "(Prediction.run)'method' arg is not implemented, change prediction method"
            )
        return predictions
