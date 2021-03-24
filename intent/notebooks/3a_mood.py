# %% [markdown]
## VP intent parsing
#
# author: Steeve Laquitaine
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
## Detect multi-sentences queries
# %% [markdown]
## Detect Moods
# %%
sent_moods = features.classify_sentence_type(cfg["text"])
# %% [markdown]
# %%
posting_list = retrieval.create_posting_list(tag)
sim_ranked = similarity.rank_nearest_to_seed(sim_matx, seed=SEED)
ranked = similarity.print_ranked_VPs(cfg, posting_list, sim_ranked)
filtered = ranked[ranked["score"] >= THRES]
# %% [markdown]
## INTENT PARSING
#
# * Apply dependency parsing to each query
# * Collect intent's action (ROOT) and object (dobj)
# %%
intents = parsing.parse_intent(filtered)
intents
