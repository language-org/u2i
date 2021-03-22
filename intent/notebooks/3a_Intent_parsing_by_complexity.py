# %% [markdown]
## Intent parsing by complexity
#
# author: Steeve Laquitaine
#
# Complexity:
#
#   * single vs. multi-sentences
#   * single vs. multi-clauses
#
# TABLE OF CONTENTS
#
# %% [markdown]
## PACKAGES
# %%
import os
from collections import defaultdict

import pandas as pd
import spacy
from matplotlib import pyplot as plt

# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from intent.src.intent.nodes import features, parsing, retrieval, similarity

# %% [markdown]
## PATHS
# %%
cfg_path = (
    proj_path + "intent/data/02_intermediate/cfg_25_02_2021_18_16_42.xlsx"
)
features_path = proj_path + "intent/data/04_feature/features.xlsx"
sim_path = proj_path + "intent/data/02_intermediate/sim_matrix.xlsx"
tag_path = proj_path + "intent/data/02_intermediate/tag.xlsx"
# %% [markdown]
## PARAMETERS
# %%
SEED = " VB NP"  # seed for comparison
THRES = 0  # keep query syntaxes > this similarity thresh
# %% [markdown]
# LOAD DATA
cfg = pd.read_excel(cfg_path)
tag = pd.read_excel(tag_path)
sim_matx = pd.read_excel(sim_path)
# %% [markdown]
## Detect multi-sentence queries
# %%
counts = features.count(cfg["text"])
cfg["sent_count"] = counts
# %%
fig = plt.figure(figsize=(5, 5))
ax = cfg["sent_count"].hist(width=0.8)
ax.grid(False)
xl = plt.xlabel("Sentence in query (count)")
yl = plt.ylabel("Occurrence (count)")
# %% [markdown]
## WRITE
# %%
cfg.to_excel(features_path)

