# %% [markdown]
## Convert a text to a graph
# author: Steeve Laquitaine
#
# TABLE OF CONTENTS
#
# Packages
# Load data
# Parameters
# queries to verb phrases
# %% [markdown]
# PACKAGES
# %%
import init
import pandas as pd
from matplotlib import pyplot as plt
from networkx import nx

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
# pos = nx.spring_layout(graph)
pos = nx.spiral_layout(graph)
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
# to convert to notebook
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb
# %% [markdown]
# # References
#
# (1) https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
#
# (2) https://stackoverflow.com/questions/32441605/generating-ngrams-unigrams-bigrams-etc-from-a-large-text-of-txt-files-and-t
#
# (3) https://github.com/Nath-B/Graph-Of-Words
#
# (4) Fragkiskos D Malliaros and Michalis Vazirgiannis, “Clustering and Community Detection in Directed Networks: A Survey,” Physics Reports 533, no. 4 (2013): 95–142.
