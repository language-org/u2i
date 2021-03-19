# %% [markdown]
## VP graph similarity
#
# author: Steeve Laquitaine
#
# TABLE OF CONTENTS
#
# * Packages
# * Paths
# * Text to VP graph
# * VP graph similarity
#   * Edit distance
#   * Jaccard distance
#   * Longest common contiguous subsequence (LCS)
# * VP graph clustering
# * Interpretation

# %% [markdown]
# PACKAGES
# %%
import os
from difflib import SequenceMatcher

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import graphs, parsing, similarity

to_df = pd.DataFrame
to_series = pd.Series
# %% [markdown]
## PATHS
# %%
cfg_path = (
    proj_path + "intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx"
)
sim_path = proj_path + "intent/data/02_intermediate/sim_matrix.xlsx"
tag_path = proj_path + "intent/data/02_intermediate/tag.xlsx"
# load dataset containing context free grammar productions
cfg = pd.read_excel(cfg_path)

# %% [markdown]
## TEXT TO GRAPH
# %%
# convert each Verb Phrase to a single graph
tag = parsing.from_cfg_to_constituents(cfg["cfg"])
vp_graph = [
    graphs.from_text_to_graph(to_series(vp), isdirected=True, isweighted=True)
    for vp in tag.to_list()
]
# %% [markdown]
## VP GRAPH SIMILARITY
# %% [markdown]
### Edit distance
#
# Calculate the total edit operation cost needed to make two graphs isomorphic.
# %%
# calculate (TODO: speed up)
ged = similarity.calc_ged(vp_graph)
# %%
# preview sample cluster VP productions (hierar. clustering)
fig = plt.figure(figsize=(10, 10))
n_sample = 20
sample_ged_df = pd.DataFrame(
    ged[:n_sample, :n_sample], index=tag[:n_sample], columns=tag[:n_sample],
)
cm = sns.clustermap(
    sample_ged_df,
    row_cluster=False,
    method="average",
    linewidths=0.15,
    figsize=(12, 13),
    cmap="YlOrBr",
    annot=ged[:n_sample, :n_sample],
)
# %%
# Cluster VP productions (hierar. clustering)
fig = plt.figure(figsize=(10, 10))
ged_df = to_df(ged, index=tag, columns=tag)
cm = sns.clustermap(
    ged_df,
    row_cluster=False,
    method="average",
    linewidths=0.15,
    figsize=(12, 13),
    cmap="vlag",
)
# %%
fig = plt.figure(figsize=(10, 2))
dist = sns.distplot(ged, bins=len(np.unique(ged)))
# %% [markdown]
#
# **Graph ('VB','NP') and ('VBD','VP') are `isomorphic`, thus GED is 0**. <br>
# This is not what we look for: we look for distance to `automorphism`. <br>
# [TODO]: consider "minimum common subgraph" distance (10)
# %%
nx.is_isomorphic(vp_graph[0], vp_graph[1])
# %% [markdown]
# ideas
# * sorting productions should group similar productions together
#
# %% [markdown]
### Jaccard distance
# [TODO]: implement
# %% [markdown]
### Longest common (contiguous) subsequence (LCS)
#
# [TODO]: implement the efficient suffix tree algo instead of the dynamic programming one
# %%
# calculate lcs similarity matrix
n_query = len(tag)
lcs_sim = np.zeros((n_query, n_query))
for ix in range(n_query):
    for jx in range(n_query):
        lcs_sim[ix, jx] = similarity.calc_lcs(tag[ix], tag[jx])

# %%
# preview sample cluster VP productions (hierar. clustering)
fig = plt.figure(figsize=(10, 10))
n_sample = 20
sample_lcs_df = pd.DataFrame(
    lcs_sim[:n_sample, :n_sample],
    index=tag[:n_sample],
    columns=tag[:n_sample],
)
cm = sns.clustermap(
    sample_lcs_df,
    row_cluster=False,
    method="average",
    linewidths=0.15,
    figsize=(12, 13),
    cmap="YlOrBr",
    annot=lcs_sim[:n_sample, :n_sample],
)
# %%
# Cluster VP productions based on lcs similarity (hierar. clustering)
fig = plt.figure(figsize=(10, 10))
lcs_sim_df = to_df(lcs_sim, index=tag, columns=tag)
cm = sns.clustermap(
    lcs_sim_df,
    row_cluster=False,
    method="average",
    linewidths=0.15,
    figsize=(12, 13),
    cmap="vlag",
)
# %% [markdown]
# Write output
# [TODO] better design
# %%
lcs_sim_df.drop_duplicates().T.drop_duplicates().to_excel(sim_path)
tag.to_excel(tag_path)
# %% [markdown]
## References
#
# (6) https://en.wikipedia.org/wiki/Louvain_method <br>
# (5) https://cdlib.readthedocs.io/en/latest/reference/generated/cdlib.viz.plot_network_clusters.html#cdlib.viz.plot_network_clusters <br>
# (4) Fragkiskos D Malliaros and Michalis Vazirgiannis, “Clustering and Community Detection in Directed Networks: A Survey,” Physics Reports 533, no. 4 (2013): 95–142. <br>
# (3) https://github.com/Nath-B/Graph-Of-Words <br>
# (2) https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-text-of-txt-files-and-t  <br>
# (1) https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html <br>
# (7) https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html <br>
# (8) https://stackoverflow.com/questions/38705359/how-to-give-sns-clustermap-a-precomputed-distance-matri <br>
# (9) https://stackoverflow.com/questions/63927196/networkx-how-to-set-custom-cost-function <br>
# (10) https://stellargraph.readthedocs.io/en/stable/demos/embeddings/gcn-unsupervised-graph-embeddings.html <br>
# (11) https://stackoverflow.com/questions/61421491/similarity-measure-between-graphs-using-networkx
# (12) https://towardsdatascience.com/sequencematcher-in-python-6b1e6f3915fc
# (13) https://gmarti.gitlab.io/ml/2017/09/07/how-to-sort-distance-matrix.html
# %% [markdown]
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb
# to convert to notebook
# %% [markdown]
