# author: steeve LAQUITAINE

# import packages
import os
from difflib import SequenceMatcher

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

# set project path
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

# import custom nodes
from intent.src.intent.nodes import graphs, parsing, similarity

# load path catalog
with open(proj_path + "intent/conf/base/catalog.yml") as file:
    catalog = yaml.load(file)

# set shortcuts
to_df = pd.DataFrame
to_series = pd.Series


class Lcs:
    """Longest common subsequence calculation class
    Calculates the similarity of syntax between query pairs
    """

    def __init__(self, verbose: bool = False):
        """Instantiates class"""
        self.sim_path = catalog["sim"]
        # try:
        #     # load intents' syntactical features (cfg)
        #     self.cfg = pd.read_excel(catalog["cfg"])
        # except:
        #     raise FileNotFoundError("(Lcs) The file path was not found.")
        # if verbose:
        #     print(f"(Lcs) The loaded cfg path is: {self.sim_path}")

    # def do(self, verbose: bool = False) -> pd.DataFrame:
    #     """Calculate intent similarity matrix

    #     Args:
    #         verbose (bool, optional): [description]. Defaults to False.

    #     Returns:
    #         pd.DataFrame: [description]
    #     """

    #     # convert each Verb Phrase to a graph
    #     tag = parsing.from_cfg_to_constituents(self.cfg["cfg"])

    #     # [TODO]: implement "suffix tree algo", it is more efficient
    #     # instead of currently used dynamic programming
    #     # calculate intent syntax' similarity matrix
    #     n_query = len(tag)
    #     lcs = np.zeros((n_query, n_query))
    #     for ix in range(n_query):
    #         for jx in range(n_query):
    #             lcs[ix, jx] = similarity.calc_lcs(tag[ix], tag[jx])
    #     lcs_df = to_df(lcs, index=tag, columns=tag)

    #     from ipdb import set_trace

    #     set_trace()

    #     if verbose:

    #         # show sample clustered verb phrase cfg
    #         # (hierar. clustering)
    #         fig = plt.figure(figsize=(10, 10))
    #         n_sample = 20
    #         sample = pd.DataFrame(
    #             lcs[:n_sample, :n_sample],
    #             index=tag[:n_sample],
    #             columns=tag[:n_sample],
    #         )

    #         cm = sns.clustermap(
    #             sample,
    #             row_cluster=False,
    #             method="average",
    #             linewidths=0.15,
    #             figsize=(12, 13),
    #             cmap="YlOrBr",
    #             annot=lcs_df[:n_sample, :n_sample],
    #         )

    #         # Cluster verb phrases' cfg (hierar. clustering)
    #         fig = plt.figure(figsize=(10, 10))
    #         cm = sns.clustermap(
    #             lcs_df,
    #             row_cluster=False,
    #             method="average",
    #             linewidths=0.15,
    #             figsize=(12, 13),
    #             cmap="vlag",
    #         )

    #     # drop syntax duplicates (duplicated indices and columns)
    #     sim = lcs_df[~lcs_df.index.duplicated(keep="first")]
    #     sim = sim.loc[:, ~sim.columns.duplicated()]

    #     # write
    #     sim.to_excel(self.sim_path)
    #     return sim

    def do(self, cfg: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """Calculate similarity matrix

        Args:
            verbose (bool, optional): [description]. Defaults to False.

        Returns:
            pd.DataFrame: [description]
        """

        # convert each Verb Phrase to a graph
        tag = parsing.from_cfg_to_constituents(cfg["cfg"])

        # [TODO]: implement "suffix tree algo", it is more efficient
        # instead of currently used dynamic programming
        # calculate intent syntax' similarity matrix
        n_query = len(tag)
        lcs = np.zeros((n_query, n_query))
        for ix in range(n_query):
            for jx in range(n_query):
                lcs[ix, jx] = similarity.calc_lcs(tag[ix], tag[jx])
        lcs_df = to_df(lcs, index=tag, columns=tag)

        if verbose:

            # show sample clustered verb phrase cfg
            # (hierar. clustering)
            fig = plt.figure(figsize=(10, 10))
            n_sample = 20
            sample = pd.DataFrame(
                lcs[:n_sample, :n_sample],
                index=tag[:n_sample],
                columns=tag[:n_sample],
            )

            cm = sns.clustermap(
                sample,
                row_cluster=False,
                method="average",
                linewidths=0.15,
                figsize=(12, 13),
                cmap="YlOrBr",
                annot=lcs_df[:n_sample, :n_sample],
            )

            # Cluster verb phrases' cfg (hierar. clustering)
            fig = plt.figure(figsize=(10, 10))
            cm = sns.clustermap(
                lcs_df,
                row_cluster=False,
                method="average",
                linewidths=0.15,
                figsize=(12, 13),
                cmap="vlag",
            )

        # drop syntax duplicates (duplicated indices and columns)
        sim = lcs_df[~lcs_df.index.duplicated(keep="first")]
        sim = sim.loc[:, ~sim.columns.duplicated()]

        # write
        sim.to_excel(self.sim_path)
        return sim


class Ged:
    """Similarity based on graph edit distance"""

    def __init__(self):
        self.sim_path = catalog["sim"]
        self.cfg = pd.read_excel(catalog["cfg"])

    def do(self):

        # convert each verb phrase to a graph
        tag = parsing.from_cfg_to_constituents(self.cfg["cfg"])
        vp_graph = [
            graphs.from_text_to_graph(to_series(vp), isdirected=True, isweighted=True)
            for vp in tag.to_list()
        ]

        # Calculate the total edit operation cost needed to make two graphs isomorphic.
        # [TODO]: speed up
        return similarity.calc_ged(vp_graph)


class Jaccard:
    """Jaccard similarity"""

    pass
