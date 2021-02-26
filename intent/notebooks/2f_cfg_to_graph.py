# %% [markdown]
## CFG in .csv to graph

# %%
import os

import networkx as nx
import pandas as pd

# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import graphs, parsing

# %% [markdown]
## LOAD CFG DATA
# %%
cfg_path = (
    proj_path + "intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx"
)
data = pd.read_excel(cfg_path)
# %% [markdown]
## PARAMETERS
# %%from intent.src.intent.nodes import graphs
isdirected = False
isweighted = False
size_window = 2  # bigrams
# %% [markdown]
## EXTRACT VP CFG right side production
# %%
constituents = parsing.from_cfg_to_constituents(data["cfg"])
# %% [markdown]
## TEXT TO GRAPH
# %%
# %%
graph = graphs.from_text_to_graph(
    constituents,
    isdirected=isdirected,
    isweighted=isweighted,
    size_window=size_window,
)
# %% [markdown]
# **Fig. text graph in different configurations**
# %% [markdown]
# **Spring**
# %%
nx.draw(
    graph,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %% [markdown]
# **Circular**
# %%
pos = nx.circular_layout(graph)
nx.draw(
    graph,
    pos=pos,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %% [markdown]
# **kamada_kawai**
# %%
pos = nx.kamada_kawai_layout(graph)
nx.draw(
    graph,
    pos=pos,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %% [markdown]
# **spring**
# %%
pos = nx.spring_layout(graph)
nx.draw(
    graph,
    pos=pos,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %% [markdown]
# **spectral**
# %%
pos = nx.spectral_layout(graph)
nx.draw(
    graph,
    pos=pos,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %% [markdown]
# **spectral**
# %%
pos = nx.spiral_layout(graph)
nx.draw(
    graph,
    pos=pos,
    node_color="w",  # node facecolor
    edgecolors="k",  # node boundary's color (3)
    linewidths=1,
    with_labels=True,
    font_weight="bold",
)
# %%
