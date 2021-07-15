# %% [markdown]
# # Intent inference
# %% [markdown]
# author: Steeve Laquitaine
# %% [markdown]
# # Summary
# %% [markdown]
# purpose:
#
#   - predict intent class
#
# approach:
#
#   - preprocessing:
#
#       - Constituency parsing
#       - Filtering:
#
#           - complexity: keep intents w/ N sentences
#           - mood
#           - syntax similarity
#
#   - model:
#
#       - model intents' similarity and cluster intents
#
#   - inference:
#
#       - predict cluster labels
#
# Observations:
#
#   * Current best parameters are:
#
#       SEED            = " VB NP" <br>
#       THRES_NUM_SENT  = 1 <br>
#       NUM_SENT        = 1 <br>
#       THRES_SIM_SCORE = 1 <br>
#       FILT_MOOD       = ("ask",) <br>
#
#
# [TODO]:
#
#  - Constituency parsing:
#       - Allennlp produces warning when no internet connection: "Connection error occurred while trying to fetch ETag for https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz. Will attempt to use latest cached version of resource"
# - add Logging
# - check that data are not created that overload memory

# %% [markdown]
# # Setup
# %% [markdown]
# ## Packages
# %%
# set project path
from inspect import TPFLAGS_IS_ABSTRACT
import os
from collections import defaultdict
from time import time

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

# import packages
import pandas as pd
import spacy
import yaml
from collections import Counter
import numpy as np

# import custom packages
from intent.src.intent.nodes import (
    model,
)
from intent.src.intent.nodes.processing import Processing
from intent.src.intent.nodes.inference import Prediction
from intent.src.intent.nodes.evaluation import calculate_accuracy

from intent.src.intent.pipelines.similarity import Lcs
from intent.src.tests import test_run
from intent.src.tests import test_run
from intent.src.intent.nodes.evaluation import calculate_accuracy

# %% [markdown]
# ## Display
pd.set_option(
    "display.max_rows",
    None,
    "display.max_columns",
    None,
    "display.width",
    None,
    "display.max_colwidth",
    None,
)
# %% [markdown]
## Parameters
# %%
with open(proj_path + "intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)
# %% [markdown]
## Loading data
# %%
t0 = time()
corpus_path = proj_path + "intent/data/01_raw/banking77/train.csv"
corpus = pd.read_csv(corpus_path)
# %% [markdown]
# # Processing
# %%
processed = Processing(
    params=prms,
    num_sent=prms["NUM_SENT"],
    filt_mood=prms["FILT_MOOD"],
    thres_sim_score=prms["THRES_SIM_SCORE"],
    seed=prms["SEED"],
).run(corpus)
# %% [markdown]
## Modeling
# %%
clustered = model.cluster_queries(
    processed, dist_thresh=prms["DIST_THRES"], hcl_method=prms["HCL_METHOD"]
)
# %% [markdown]
## Inference
# %%
with_predictions = Prediction(method=prms["PREDICT_METHOD"]).run(corpus, clustered)
# %% [markdown]
## Evaluation
# %%
accuracy = calculate_accuracy(
    with_predictions["true_labels"], with_predictions["predicted"]
)
