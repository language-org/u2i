# %% [markdown]
## INTENT PARSING
# %% [markdown]
# * **Purpose** :
#   * Test intent parsing with ALLENLP
# %% [markdown]
## TABLE OF CONTENT
##SETUP
##PARAMETERS
##PARSING
### Allennlp
### VP extraction
# %%[markdown]
# # SETUP
# %%
import os
from time import time

import numpy as np
import pandas as pd
from nltk.tree import ParentedTree
from pigeon import annotate

from intent.src.intent.nodes import mood, parsing

pd.set_option("display.max_colwidth", 100)
# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
tr_data_path = proj_path + "data/01_raw/banking77/train.csv"
test_data_path = proj_path + "data/01_raw/banking77/test.csv"
os.chdir(proj_path)
# %% [markdown]
# # PARAMETERS
prm = dict()
prm["sample"] = 100
prm["mood"] = ["declarative"]
# prm[
#     "intent_class"
# ] = "card_arrival"  # "contactless_not_working"  # small class with 35 samples
prm["intent_class"] = "contactless_not_working"  # small class with 35 samples
# %%
# read queries data
tr_data = pd.read_csv(tr_data_path)
# %%
# select data for an input class
data_class_i = tr_data[tr_data["category"].eq(prm["intent_class"])]
# %%
data_class_i.head(5)
# %%
sample = data_class_i["text"].iloc[0]
# %% [markdown]
# # PARSING
# %% [markdown]
# ## ALLENLP
# %%
tic = time()
al_prdctor = parsing.init_allen_parser()
print(f"(Instantiation) took {round(time()-tic,2)} secs")
# %%
tic = time()
output = al_prdctor.predict(sentence=sample)
parsed_txt = output["trees"]
print(f"(Inference) took {round(time()-tic,2)} secs")
print(f"Parsed sample:\n{parsed_txt}")
# %% [markdown]
# ## VP EXTRACTION
# %%
tree = ParentedTree.fromstring(parsed_txt)
assert len(parsing.extract_VP(al_prdctor, "I want coffee")) > 0, "VP is Empty"
# %%
# Speed up (1 hour / 10K queries)
VPs = parsing.extract_all_VPs(prm, data_class_i, al_prdctor)
assert (
    len(VPs) == len(data_class_i) or len(VPs) == prm["sample"]
), '''VP's length does not match "data_class_i"'''
# %%
# add to data, show
data_class_i["VP"] = pd.DataFrame(VPs)
data_class_i.iloc[: prm["sample"]]
# %% [markdown]
## ANNOTATE
# %%
annots = annotate(data_class_i["VP"], options=["yes", "no"])
# %%
# verb_p[0].pretty_print()
# %% [markdown]
## SELECT QUERY W/ INPUT MOOD
# TODO:
# 1. Annotate VPs that look like intent vs. not
# 2. Look what make them different
# 3. Test some hypothesis:
#   - mood: declarative vs. interrogative syntax ?
#   - tense: present vs. past ?
#   - lexical: some verbs and not others
#   - else ?
#   - semantics: direct object vs. indirect ?
# %%
moods = mood.classify_sentence_type(data_class_i["text"])
moods
