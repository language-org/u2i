# %%
import os
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

from intent.src.intent.nodes import similarity

# %% [markdown]
## DATA
# %%
text = (
    "drink coffee",
    "drink tea",
    "drink water",
    "travel to France",
    "journey to Italy",
    "buy a house",
    "buy a building",
)
# %% [markdown]
## PARAMETERS
#
# distance threshold t - the maximum inter-cluster distance allowed
# %%
DIST_THRES = 1.2
#%%
clusters = similarity.get_semantic_clusters(text, DIST_THRES, verbose=True)
clusters.sort_values(by=["cluster"])
