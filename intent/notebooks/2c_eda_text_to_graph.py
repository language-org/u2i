# %% [markdown]
## Convert a text to a graph
# author: Steeve Laquitaine
#
# %% [markdown]
# PACKAGES
# %%
import init
import pandas as pd
from networkx import nx

from intent.src.intent.nodes import graphs

# %% [markdown]
## LOAD DATA
# %%
text = pd.Series(
    ["I would like to understand the mind.", "I want to do good research."]
)
# %% [markdown]
## PARAMETERS
# %%
isdirected = False
isweighted = False
size_window = 2  # bigrams
# %% [markdown]
## TEXT TO GRAPH
# %%
# %%
graph = graphs.from_text_to_graph(
    text, isdirected=isdirected, isweighted=isweighted, size_window=size_window
)
# %% [markdown]
# **Fig. text graph**
nx.draw(graph, with_labels=True, font_weight="bold")
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


# %%
