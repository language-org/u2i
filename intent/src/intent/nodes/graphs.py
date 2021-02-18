# author: steeve laquitaine

import networkx as nx
import pandas as pd


def get_graph_from_document(doc_str:str, isdirected:bool, isweighted:bool, size_window:int):
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
        G=nx.Graph()
        
    for j in range(N):
        for i in range(max(j-size_window+1,0),j):
            if G.has_edge(doc_array[i], doc_array[j]):
                if isweighted:
                    # we added this one before, just increase the weight by one
                    G[doc_array[i]][doc_array[j]]['weight'] += 1
            else:
                # new edge. add with weight=1
                G.add_edge(doc_array[i], doc_array[j], weight=1)

    return G

def get_gow(corpus:pd.Series, isdirected:bool, isweighted:bool, size_window:int):
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
         dict_graph_of_words[i] = get_graph_from_document(corpus[i],isdirected,isweighted, size_window)
        
    return dict_graph_of_words

