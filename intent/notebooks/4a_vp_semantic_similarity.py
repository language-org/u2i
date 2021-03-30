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
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
from nltk.corpus import wordnet as wn

from intent.src.intent.nodes import inference, preprocess

# %% [markdown]
## DATA
# %%
corpus = (
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
## PREPROCESSING
#
# * Mispelled: drop mispelled words [TODO]: make more efficient
# We filter out all words not contained in Wordnet
# %%
filtered_corpus = preprocess.filter_words_not_in_wordnet(corpus)
# %% [markdown]
## INFERENCE
# %%
df = inference.label_queries(filtered_corpus, DIST_THRES)
df
# %% [markdown]
#
# (1) https://stackoverflow.com/questions/13555399/nltk-wordnet-similarity-returns-none-for-adjectives
