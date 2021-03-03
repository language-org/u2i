# %% [markdown]
## Search clusters in graph
# author: Steeve Laquitaine
#
# TABLE OF CONTENTS
#
# Packages <br>
# Load data <br>
# Parameters <br>
# Queries to verb phrases <br>
# Verb phrases to graph <br>
# Search clusters with LOUVAIN <br>
# %% [markdown]
# PACKAGES
# %%

from itertools import chain, repeat

import init
import networkx as nx
import pandas as pd
from cdlib import algorithms, viz
from matplotlib import pyplot as plt

from intent.src.intent.nodes import graphs, parsing

# %% [markdown]
## LOAD DATA
# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
raw_data_path = proj_path + "intent/data/01_raw/banking77/train.csv"

data = pd.read_csv(raw_data_path)
# %% [markdown]
## PARAMETERS
# %%
isdirected = True
isweighted = True
size_window = 2  # bigrams
# %% [markdown]
## QUERIES TO VERB PHRASES
# %%
al_prdctor = parsing.init_allen_parser()
verb_phrases_str = parsing.from_text_to_constituents(data, al_prdctor)
# %% [markdown]
## VERB PHRASES TO GRAPH
# %%
graph = graphs.from_text_to_graph(
    verb_phrases_str,
    isdirected=isdirected,
    isweighted=isweighted,
    size_window=size_window,
)
# %% [markdown]
# **Fig. text graph**
# %%
fig = plt.figure(figsize=(10, 10))
pos = nx.kamada_kawai_layout(graph)
nx.draw(
    graph,
    pos=pos,
    node_size=1500,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %% [markdown]
# # Search clusters with LOUVAIN
#
# * LOUVAIN is a node clustering heuristics that maximizes modularity
# (such that cluster have more internal connections that external)
# * It has O(n * log2n) complexity (6)
# * It does not use directionality to identify clusters
#
# Drop graph directionality for clustering and search clusters <br>
# Plot the directed graph with the undirected graph's node clusters represented by different colors
# %%
undirected_graph = graph.to_undirected()
communities = algorithms.louvain(
    undirected_graph, weight="weight", resolution=1.0, randomize=False
)
# %%
# plot with networkx (more control)
# assign colors to clusters
node_colors = [
    list(zip(nodes, repeat(colr, len(nodes))))
    for colr, nodes in enumerate(communities.communities)
]
node_colors = dict(chain.from_iterable(node_colors))
nx.set_node_attributes(graph, node_colors, "color")

# plot
fig = plt.figure(figsize=(10, 10))
nx.draw(
    graph,
    pos=pos,
    node_color=list(node_colors.values()),
    node_size=1500,
    edgecolors="k",
    linewidths=1,
    with_labels=True,
    font_weight="bold",
    cmap=plt.cm.Set2,
)
# %% [markdown]
# to convert to notebook
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb
# %% [markdown]
# # References
#
# (1) https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html <br>
# (2) https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-text-of-txt-files-and-t  <br>
# (3) https://github.com/Nath-B/Graph-Of-Words <br>
# (4) Fragkiskos D Malliaros and Michalis Vazirgiannis, “Clustering and Community Detection in Directed Networks: A Survey,” Physics Reports 533, no. 4 (2013): 95–142. <br>
# (5) https://cdlib.readthedocs.io/en/latest/reference/generated/cdlib.viz.plot_network_clusters.html#cdlib.viz.plot_network_clusters <br>
# (6) https://en.wikipedia.org/wiki/Louvain_method <br>

# %%
