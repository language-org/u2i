# %% [markdown]
# # INTENT PARSING
# %% [markdown]
# * **Purpose** :
#  * Test constituency parsers to extracy Verb Phrases
# %%[markdown]
# # SETUP
# %%
import os

proj_path = "/Users/steeve_laquitaine/Desktop/CodeHub/intent/intent"
os.chdir(proj_path)

import pandas as pd
import time
import numpy as np
from time import time

# visualization
from matplotlib import pyplot as plt

# preprocessing
import spacy
from spacy import tokenizer
from spacy.lang.en import English

# exploration
import re
from ipywidgets import interact

# nlp
from src.intent.nodes import parsing
import allennlp

# display
pd.set_option("display.max_colwidth", 100)
# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
train_data_path = proj_path + "data/01_raw/banking77/train.csv"
test_data_path = proj_path + "data/01_raw/banking77/test.csv"
# %%
train_data = pd.read_csv(train_data_path)
# %%
train_data.head(5)
# %%
sample = train_data["text"].iloc[0]
# %% [markdown]
# # PARSING
# %% [markdown]
# ## ALLENLP
# %%
tic = time()
allen_predictor = parsing.instantiate_allennlp_constituency_parser()
print(f"(Instantiation) took {round(time()-tic,2)} secs")
# %%
tic = time()
output = allen_predictor.predict(sentence=sample)
all_parsed_sample = output["trees"]
print(f"(Inference) took {round(time()-tic,2)} secs")
print(f"Parsed sample:\n{all_parsed_sample}")
# %% [markdown]
# ## STANZA CORENLP
# [TODO]: change all paths in nodes.parsing.py
# %%
tic = time()
parsing.setup_stanza()
print(f"(Setup) took {round(time()-tic,2)} secs")
# %%
tic = time()
cp_parser = parsing.init_Stanza_constituency_parsing(sample)
print(f"(Instantiation) took {round(time()-tic,2)} secs")
# %%
tic = time()
stz_parsed_sample = cp_parser.sentence[0].parseTree
print(f"(Inference) took {round(time()-tic,5)} secs")
print(f"Parsed sample:\n{stz_parsed_sample}")
# %%
print("Complete")

# %%
