# author: steeve laquitaine

from difflib import SequenceMatcher
from itertools import chain, repeat

import networkx as nx
import numpy as np
import pandas as pd


# graph edit distance's cost functions
def node_subst_cost(node1, node2):
    if node1 == node2:
        return 0
    return 1


def node_del_cost(node):
    return 1


def node_ins_cost(node):
    return 1


def edge_subst_cost(edge1, edge2):
    if edge1 == edge2:
        return 0
    return 1


def edge_del_cost(node):
    return 1  # here you apply the cost for edge deletion


def edge_ins_cost(node):
    return 1  # here you apply the cost for edge insertion


def calc_ged(graphs_of_VPs: list):
    """Calculate graph edit distance

    Args:
        graphs_of_VPs (list): list of networkx graphs  

    Returns:
        [np.array]: matrix of pairwise graph edit distances
    """
    n_graphs = len(graphs_of_VPs)

    ged_sim = np.zeros((n_graphs, n_graphs))
    for ix in range(n_graphs):
        for jx in range(n_graphs):
            ged_sim[ix, jx] = nx.graph_edit_distance(
                graphs_of_VPs[ix],
                graphs_of_VPs[jx],
                node_subst_cost=node_subst_cost,
                node_del_cost=node_del_cost,
                node_ins_cost=node_ins_cost,
                edge_subst_cost=edge_subst_cost,
                edge_del_cost=edge_del_cost,
                edge_ins_cost=edge_ins_cost,
            )
    return ged_sim


def calc_lcs(str1: str, str2: str) -> float:
    """Calculate the length ratio of the longest common subsequence between two strings

    Args:
        str1 (str): string to match
        str2 (str): string to match

    Returns:
        float: length ratio of the longest common subsequence b/w str1 and str2
    """
    s = SequenceMatcher(None, str1, str2)
    match = s.find_longest_match(0, len(str1), 0, len(str2))
    match_content = str1[match.a : match.size]
    lcs_similarity = s.ratio()
    return lcs_similarity


def print_ranked_VPs(
    cfg: pd.DataFrame, posting_list, sorted: pd.Series
) -> pd.Series:
    """Rank verb phrases by syntactic similarity score

    Args:
        cfg (pd.DataFrame): context free grammar production rules (VP -> VB NP)
        posting_list (defauldict): list of position indices for the production right side (e.g., VB NP)
        sorted (pd.Series): [description]

    Returns:
        pd.Series: syntactic similarity score b/w seed query and other queries
    """
    index = list(
        chain.from_iterable(
            [posting_list[sorted.index[ix]] for ix in range(len(sorted))]
        )
    )
    score = list(
        chain.from_iterable(
            [
                list(repeat(sorted[ix], len(posting_list[sorted.index[ix]])))
                for ix in range(len(sorted))
            ]
        )
    )
    ranked_vps = cfg["VP"].iloc[index]
    df = pd.DataFrame(ranked_vps, columns=["VP"])
    df["score"] = score
    return df


def rank_nearest_to_seed(simil_matx: pd.DataFrame, seed: str) -> pd.Series:
    """rank by similarity to seed syntax

    Args:
        simil_matx (pd.DataFrame): queries syntax similarity matrix  
        seed (str): syntax seed
            e.g., 'VB NP'   

    Returns:
        pd.Series: queries' syntax ranked in descending order of similarity to seed
    """
    dedup = (
        simil_matx[simil_matx["cfg"].eq(seed)]
        .drop_duplicates()
        .T.drop_duplicates()
        .T
    )
    sim_ranked = dedup.iloc[0, 1:].sort_values(ascending=False)
    return sim_ranked
