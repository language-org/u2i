# %% [markdown]
## EDA VP GRAPHS
# author: Steeve Laquitaine
#
# %% [markdown]
#* **Purpose** :
#  * Find clusters of VPs characteristic of well-formed intent queries 
#  * Extract feature signatures for well-formed intent clusters  
# %% [markdown]
# * TABLE OF CONTENT  
# * GRAPHS of VP's constituents
#
# TODO: svg plots can't be converted to pdf. Convert to png.
# %% [markdown]
## GRAPHS of VP constituent connections
# 
# Description: each query is represented as a trajectory or path in a graph
#
# Hypothesis: well-formed intent VPs are stereotyped paths which form clusters over
#   the query dataset.
#
# %%
# faster for many ngrams than nltk.util.ngram
# from itertools import chain


# def n_grams(seq, n=1):
#     """Returns an iterator over the n-grams given a list_tokens"""
#     shift_token = lambda i: (el for j,el in enumerate(seq) if j>=i)
#     shifted_tokens = (shift_token(i) for i in range(n))
#     tuple_ngrams = zip(*shifted_tokens)
#     return tuple_ngrams # if join in generator : (" ".join(i) for i in tuple_ngrams)  

# def range_ngrams(list_tokens, ngram_range=(1,2)):
#     """Returns an itirator over all n-grams for n in range(ngram_range) given a list_tokens."""
#     return chain(*(n_grams(list_tokens, i) for i in range(*ngram_range)))

# input_list = input_list = 'test the ngrams generator'.split()
# edge_list = list(range_ngrams(input_list, ngram_range=(2,3)))
# from IPython.display import SVG
# # %%
# from sknetwork.utils import edgelist2adjacency
# from sknetwork.visualization import svg_digraph

# # edge_list = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]

# adjacency = edgelist2adjacency(edge_list)
# image = svg_digraph(adjacency)
# #svg_digraph(adjacency, position, names)
# SVG(image)
# %%

import init
import networkx as nx
import pandas as pd

from intent.src.intent.nodes import graphs

# %% [markdown]
# # PARAMETERS
# %%
isdirected = True
isweighted = True
size_window = 2 # bigrams
# %% [markdown]
# # GRAPHS
# %%
text = pd.Series(['I would like to understand the mind.', 'I want to do good research.'])
dict_graph_of_words = graphs.get_gow(text, isdirected, isweighted, size_window)
# %% [markdown]
# **Fig. text sample 1**
# %%
dict_graph_of_words[0].edges
# %% [markdown]
# **Fig. text sample 2**
# %%
dict_graph_of_words[1].edges
# %% [markdown]
# **Fig. graph text sample 1**
# %%
# plot
nx.draw(dict_graph_of_words[0], with_labels=True, font_weight='bold')
# %% [markdown]
# **Fig. graph text sample 2**
# %%
nx.draw(dict_graph_of_words[1], with_labels=True, font_weight='bold')
# %% [markdown]
# H0: Randomness or Chance: The `Erdos Renyi random graph model` (4)
#   * Skewed power law degree distribution  
#   * Short average distance b/w nodes (small world)  
#   *  ...
# %%
# %%
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb
# %% [markdown]  
# # References
#
# (1) https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html    
# (2) https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-text-of-txt-files-and-t   
# (3) https://github.com/Nath-B/Graph-Of-Words  
# (4) Fragkiskos D Malliaros and Michalis Vazirgiannis, “Clustering and Community Detection in Directed Networks: A Survey,” Physics Reports 533, no. 4 (2013): 95–142.  

