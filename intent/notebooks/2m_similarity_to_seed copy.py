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
from itertools import chain
from typing import DefaultDict

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import graphs, parsing, retrieval, similarity
from intent.src.tests import test_run

to_df = pd.DataFrame
to_series = pd.Series

pd.set_option("display.max_rows", 500)

# %% [markdown]
## PATHS
# %%
cfg_path = (
    proj_path + "intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx"
)
sim_path = proj_path + "intent/data/02_intermediate/sim_matrix.xlsx"
tag_path = proj_path + "intent/data/02_intermediate/tag.xlsx"
cfg = pd.read_excel(cfg_path)
tag = pd.read_excel(tag_path)

# %% [markdown]
# LOAD DATA
simil_matx = pd.read_excel(sim_path)
test_run.simil_matx(simil_matx)

# %% [markdown]
## Create posting list for efficient query retrieval
# %%
posting_list = retrieval.create_posting_list(tag)
test_run.test_posting_list(posting_list, tag)
posting_list
# %% [markdown]
## Similarity ranking to a Seed
# %%
seed = " VB NP"
deduplicated = (
    simil_matx[simil_matx["cfg"].eq(seed)]
    .drop_duplicates()
    .T.drop_duplicates()
    .T
)
sorted = deduplicated.iloc[0, 1:].sort_values(ascending=False)
sorted
# %% [markdown]
## Display original queries ranked by similarity
# %%
df = similarity.print_ranked_VPs(cfg, posting_list, sorted)
df
