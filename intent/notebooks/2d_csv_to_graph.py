
# %% [markdown]
## CFG in .csv to graph

# %%
import os

import networkx as nx
import pandas as pd

# %%
proj_path= "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import graphs

# %% [markdown]
## LOAD CFG DATA
# %%
cfg_path = proj_path + 'intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx'
data = pd.read_excel(cfg_path)
# %% [markdown]
## PARAMETERS
# %%from intent.src.intent.nodes import graphs
isdirected = False
isweighted = False
size_window = 2 # bigrams
# %% [markdown]
## EXTRACT VP CFG right side production
# %%
cfg = data['cfg']
constituents = cfg.apply(lambda x: x.replace('VP ->', ''))

# %% [markdown]
## TEXT TO GRAPH
# %%
# %%
graph = graphs.from_text_to_graph(constituents, isdirected=isdirected, isweighted=isweighted, size_window=size_window)
# %% [markdown]
# **Fig. text graph**
nx.draw(graph, with_labels=True, font_weight='bold')
