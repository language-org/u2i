# %% [markdown]
## Label queries using Semantic similarity-based hierarchical clustering
#
# author: steeve LAQUITAINE
#
# %%
import os
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import inference

# %% [markdown]
## DATA
# %%
text = (
    "want to drink a cup of coffee",
    "want to drink a cup of tea",
    "would like a bottle of water",
    "want to track my credit card",
    "want to change my card password",
    "want to obtain a loan from the bank",
    "want to travel in a novel country",
    "want to book a trip to canada",
    "want to fly around the world",
)
# %% [markdown]
## PARAMETERS
#
# %%
# distance threshold t - the maximum inter-cluster distance allowed
DIST_THRES = 1.8
# %% [markdown]
## INFERENCE
# %%
df = inference.label_queries(text, DIST_THRES)
df
