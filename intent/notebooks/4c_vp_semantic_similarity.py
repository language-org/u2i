# %% [markdown]
## Label queries using Semantic similarity-based hierarchical clustering
#
# author: steeve LAQUITAINE
#
# * challenges:
#   * NLTK wordnet similarity
#       * returns "None" for adjectives so they must be filtered out (1)
#       * returns "None" for mispelled words not in wn.synset -> for now mispelled filtered-out
#       * returns "None" for prepositions ... which must be filtered out
#           * WordNet only contains "open-class words": nouns, verbs, adjectives, and adverbs. Thus, excluded words include determiners, prepositions, pronouns, conjunctions, and particles.
# %%
import os
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from nltk.corpus import wordnet as wn

from intent.src.intent.nodes import inference, preprocess
from intent.src.intent.pipelines.parsing import Cfg

# %% [markdown]
## PARAMETERS
# %%
SEED = " VB NP"  # seed for comparison
NUM_SENT = 1  # keep query with max one sentence
THRES_SIM_SCORE = 1  # Keep queries syntactically similar to seed
FILT_MOOD = ("ask",)  # ("state", "wish-or-excl", "ask")  # Keep statements
DIST_THRES = (
    5  # inference threshold for clustering, low values -> more clusters
)
with open(proj_path + "intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)
# %% [markdown]
## LOAD DATA
# %%
t0 = time()
corpus_path = proj_path + "intent/data/01_raw/banking77/train.csv"
corpus = pd.read_csv(corpus_path)
# %% [markdown]
## CONSTITUENCY PARSING
# %%
cfg = Cfg(corpus, prms).do()
# %%
verb_phrase = tuple(cfg["VP"])
# %% [markdown]
## PREPROCESSING
#
# * Mispelled: drop mispelled words [TODO]: make more efficient
# We filter out all words not contained in Wordnet
# %%
filtered_corpus = preprocess.filter_words_not_in_wordnet(verb_phrase)
filtered_corpus = preprocess.filter_empty_queries(filtered_corpus)
# %% [markdown]
## INFERENCE
# %%
df = inference.label_queries(filtered_corpus, DIST_THRES)
df
# %% [markdown]
#
# (1) https://stackoverflow.com/questions/13555399/nltk-wordnet-similarity-returns-none-for-adjectives

