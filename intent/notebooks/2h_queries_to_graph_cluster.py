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
# pos = nx.kamada_kawai_layout(graph)
pos = nx.spring_layout(graph)
# pos = nx.spiral_layout(graph)
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
# Drop graph directionality for clustering and search clusters <br>
# Plot the directed graph with the undirected graph's node clusters represented by different colors
# %%
undirected_graph = graph.to_undirected()
coms = algorithms.louvain(
    undirected_graph, weight="weight", resolution=1.0, randomize=False
)
# %%
pos = nx.spring_layout(graph)
v = viz.plot_network_clusters(graph, coms, pos)
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

