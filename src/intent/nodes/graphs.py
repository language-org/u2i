# author: steeve laquitaine

import networkx as nx
import pandas as pd


def get_graph_from_document(
    doc_str: str, isdirected: bool, isweighted: bool, size_window: int
):
    """Convert string text to a networkx graph object

    Args:
        doc_str (str): [description]
        isdirected (bool): [description]
        isweighted (bool): [description]
        size_window (int): [description]

    Returns:
        networkx graph object: [description]
    """

    doc_array = doc_str.split()
    N = len(doc_array)

    if isdirected:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    for j in range(N):
        for i in range(max(j - size_window + 1, 0), j):
            if G.has_edge(doc_array[i], doc_array[j]):
                if isweighted:
                    # we added this one before, just increase the weight by one
                    G[doc_array[i]][doc_array[j]]["weight"] += 1
            else:
                # new edge. add with weight=1
                G.add_edge(doc_array[i], doc_array[j], weight=1)

    return G


def get_gow(
    corpus: pd.Series, isdirected: bool, isweighted: bool, size_window: int
):
    """Convert pandas series of texts to a dictionary of networkx graph objects

    Args:
        corpus (pd.Series): [description]
        isdirected (bool): [description]
        isweighted (bool): [description]
        size_window (int): [description]

    Returns:
        dict: [description]
    """
    dict_graph_of_words = dict()

    for i in range(len(corpus)):
        dict_graph_of_words[i] = get_graph_from_document(
            corpus[i], isdirected, isweighted, size_window
        )

    return dict_graph_of_words


def from_sents_to_graphs(
    text: pd.Series,
    isdirected: bool = False,
    isweighted: bool = False,
    size_window: int = 2,
) -> dict:
    """Convert text sentences to a dictionary of networkx graphs
    note: parallelizable 

    Args:
        text (pd.Series): series of text strings
        isdirected (bool, optional): [description]. Defaults to False.
        isweighted (bool, optional): [description]. Defaults to False.
        size_window (int, optional): [description]. Defaults to 2.

    Returns:
        dict: dictionary of networkx graphs for each sentence
    """

    graphs = get_gow(text, isdirected, isweighted, size_window)

    return graphs


def from_text_to_graph(
    text: pd.Series, isdirected: bool = False, isweighted: bool = False,
) -> nx.classes.graph.Graph:
    """Convert a series of text (str) to a dictionary of networkx graphs

    Args:
        text (pd.Series): series of text strings (e.g., verb phrases chunks, full sentences,.)
        isdirected (bool, optional): [description]. Defaults to False.
        isweighted (bool, optional): [description]. Defaults to False.

    Returns:
        nx.classes.graph.Graph: dictionary of networkx graphs for each sentence
    """

    # convert each sentence to a graph
    # convert dictionary of graphs to list of graphs
    # merge all graphs
    graphs_by_sent = from_sents_to_graphs(text, isdirected, isweighted, 2)
    list_of_graphs = list(graphs_by_sent.values())
    graph = nx.compose_all(list_of_graphs)

    return graph
