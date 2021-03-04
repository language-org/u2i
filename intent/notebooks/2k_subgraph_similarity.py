# %% [markdown]
## Analyse the similarity of VPs subgraphs
# author: Steeve Laquitaine
#
# TABLE OF CONTENTS
#
# %% [markdown]
# PACKAGES
# %%
import os

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import graphs, parsing

# %%
raw_data_path = proj_path + "intent/data/01_raw/banking77/train.csv"
cfg_data_path = (
    proj_path + "intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx"
)
# load dataset containing context free grammar productions
cfg = pd.read_excel(cfg_data_path)
# %%
# convert each Verb Phrase to a single graph
constt = parsing.from_cfg_to_constituents(cfg["cfg"])
graphs_of_VPs = [
    graphs.from_text_to_graph(pd.Series(vp), isdirected=True, isweighted=True)
    for vp in constt.to_list()
]

# %% [markdown]
## Graph Edit Distance metric
# %%
# calculate similarity metrics (TODO: speed up)
n_graphs = len(graphs_of_VPs)
ged_sim = np.zeros((n_graphs, n_graphs))
for ix in range(n_graphs):
    for jx in range(n_graphs):
        ged_sim[ix, jx] = nx.graph_edit_distance(
            graphs_of_VPs[ix], graphs_of_VPs[jx]
        )
# %%
# plot similarity matrix
fig = plt.figure(figsize=(10, 10))
ax = sns.heatmap(ged_sim, fmt="f")
# %% [markdown]
## References
#
# (6) https://en.wikipedia.org/wiki/Louvain_method <br>
# (5) https://cdlib.readthedocs.io/en/latest/reference/generated/cdlib.viz.plot_network_clusters.html#cdlib.viz.plot_network_clusters <br>
# (4) Fragkiskos D Malliaros and Michalis Vazirgiannis, “Clustering and Community Detection in Directed Networks: A Survey,” Physics Reports 533, no. 4 (2013): 95–142. <br>
# (3) https://github.com/Nath-B/Graph-Of-Words <br>
# (2) https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-text-of-txt-files-and-t  <br>
# (1) https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html <br>
#
# %% [markdown]
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb
# to convert to notebook
# %% [markdown]

