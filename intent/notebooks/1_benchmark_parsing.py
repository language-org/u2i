# %% [markdown]
# # INTENT PARSING
# %% [markdown]
# * **Purpose** :
#  * Test constituency parsers to extracy Verb Phrases
# %%[markdown]
# # SETUP
# %%
import os

proj_path = "/Users/steeve_laquitaine/Desktop/CodeHub/intent/"
os.chdir(proj_path)

# exploration
import re
import time
from time import time

import allennlp
import numpy as np
import pandas as pd
# preprocessing
import spacy
from ipywidgets import interact
# visualization
from matplotlib import pyplot as plt
from nltk.tree import ParentedTree
from spacy import tokenizer
from spacy.lang.en import English

# nlp
from intent.src.intent.nodes import parsing

# display
pd.set_option("display.max_colwidth", 100)
# %% [markdown]
## DATA PATHS
# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
train_data_path = proj_path + "data/01_raw/banking77/train.csv"
test_data_path = proj_path + "data/01_raw/banking77/test.csv"
# %% [markdown]
## LOAD DATA
# %%
train_data = pd.read_csv(train_data_path)
train_data.head(5)
# %%
# sample = train_data["text"].iloc[0]
sample = "How do I track the card you sent me?"
# %% [markdown]
# # PARSING
# %% [markdown]
# ## ALLENLP
# %%
# instantiate predictor
tic = time()
allen_predictor = parsing.init_allen_parser()
print(f"(Instantiation) took {round(time()-tic,2)} secs")
# %%
# run inference
tic = time()
output = allen_predictor.predict(sentence=sample)
all_parsed_sample = output["trees"]
print(f"(Inference) took {round(time()-tic,2)} secs")
print(f"Parsed sample:\n{all_parsed_sample}")
tree = ParentedTree.fromstring(output["trees"])
print(tree)
tree
# %% [markdown]
# ## STANZA CORENLP
# [TODO]: change all paths in nodes.parsing.py
# %%
# setup
tic = time()
parsing.setup_stanza()
print(f"(Setup) took {round(time()-tic,2)} secs")
# %%
# instantiate predictor
tic = time()
cp_parser = parsing.init_Stanza_constituency_parsing(sample)
print(f"(Instantiation) took {round(time()-tic,2)} secs")
# %%
# run inference
tic = time()
stz_parsed_sample = cp_parser.sentence[0].parseTree
print(f"(Inference) took {round(time()-tic,5)} secs")
print(f"Parsed sample:\n{stz_parsed_sample}")
# %%
print("Complete")
