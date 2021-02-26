# %% [markdown]
## INTENT PARSING
# %% [markdown]
#* **Purpose** :
#  * Test intent parsing with ALLENLP
# %% [markdown]
# * TABLE OF CONTENT  
# * SETUP  
#   * paths
# * PARAMETERS  
# * PARSING 
#   * Allennlp VP parsing  
#   * Parsing performance
#   * Focus on the class well parsed   
# * ANNOTATION
#   * Annotate well-formed intent VPs vs. not
# * CFG FEATURES 
#
# To read
# * Berkeley Neural Parser w/ spacy:   
#   https://spacy.io/universe/project/self-attentive-parser    
#   https://www.analyticsvidhya.com/blog/2020/07/part-of-speechpos-tagging-dependency-parsing-and-constituency-parsing-in-nlp/  
# %%[markdown]
## SETUP
# %%
import os
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import yaml
from nltk.tree import ParentedTree
from pigeon import annotate

proj_path= "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)


# in root
from intent.src.intent.nodes import annotation, mood, parsing, preprocess
from intent.src.tests import test_run

# dataframe display
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_rows", 1000)

# pd.set_option('display.notebook_repr_html', True)

# to display df w/ nbconvert to pdf
# def _repr_latex_(self):
    # return "\centering{%s}" % self.to_latex()
# pd.DataFrame._repr_latex_ = _repr_latex_  # monkey patch pandas DataFrame
# %% [markdown]
### paths
# %%
# load catalog
with open(proj_path+"intent/conf/base/catalog.yml") as file:
    catalog = yaml.load(file)
with open(proj_path+"intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)
tr_data_path = proj_path + "intent/data/01_raw/banking77/train.csv"
test_data_path = proj_path + "intent/data/01_raw/banking77/test.csv"
# %% [markdown]
# %%
# read queries data
tr_data = pd.read_csv(tr_data_path)
# %%
sample = preprocess.sample(tr_data)
# %%
sample.head(5)
# %% [markdown]
## PARSING
# %% [markdown]
### ALLENLP VP PARSING
# %%
al_prdctor = parsing.init_allen_parser()
# %%
tic = time()
out = al_prdctor.predict(sentence=sample['text'].iloc[0])
parsed_txt = out["trees"]
print(f"(Inference) took {round(time()-tic,2)} secs")
print(f"Parsed sample:\n{parsed_txt}")
# %%
tree = ParentedTree.fromstring(parsed_txt)
test_run.test_extract_VP(al_prdctor)
# %%
# Speed up (1 hour / 10K queries)
VP_info = parsing.extract_all_VPs(sample, al_prdctor)
test_run.test_extract_all_VPs(VP_info, sample, prms)
# %%
VPs = parsing.make_VPs_readable(VP_info)
# %%
VP_info = parsing.get_CFGs(VP_info)
# %%
sample["VP"] = np.asarray(VPs)
sample["cfg"] = np.asarray([VP['cfg'] if not len(VP)==0 else None for VP in VP_info])
# %% [markdown]
# Write parsed data
# %%
sample.to_excel(catalog['parsed'])
# %%
# verb_p[0].pretty_print()
# %% [markdown]
### PARSING PERFORMANCE
#
# * **Parser works in 62% of the cases for "card_arrival" and never for other classes**
#
#   * see 2a_eda_parsing.py
#   * We will analyse why later.
#   * We now focus on the class well parsed: "card_arrival".
# %% [markdown]
### FOCUS ON THE CLASS WELL PARSED   
#
# moods = mood.classify_sentence_type(data["text"])
# moods
#
#### ANNOTATE
#
# 1. Annotate well-formed intent VPs vs. not
# %%
# can't be made into a function because of Pigeon "annotate"
if prms['annotation'] == 'do':
    annots = annotate(sample["VP"], options=["yes", "no"])
else:
    annots, myfile, myext = annotation.get_annotation(catalog, prms, sample)
# %%
annots_df = annotation.index_annots(prms, sample, annots)
# %%
annotation.write_annotation(catalog, prms, annots_df, myfile, myext)
# %%
annots_df['annots'][annots_df['VP'].isnull()] = np.nan
annots_df['text'] = sample['text']
annots_df['cfg'] = sample['cfg']
# %%
parsing.write_cfg(annots_df)
# %% [markdown]
# **Fig. Queries are sorted by annotation result below.**
# %%
sorted_annots = annots_df.sort_values(by='annots', ascending=False)
# %%
sorted_annots
# %% [markdown]
# **Fig. Only 16% of intents are well formed.**
# %%
n_total = len(sorted_annots)
n_null = sorted_annots['annots'].isnull().sum()
n_yes = sorted_annots['annots'].eq('yes').sum()
n_no = sorted_annots['annots'].eq('no').sum()
stats = pd.DataFrame({
    'annots': ['null', 'yes', 'no','Total'], 
    'count': [n_null, n_yes, n_no, n_total],
    '%': [n_null/n_total*100, n_yes/n_total*100, n_no/n_total*100, 100]
    })
stats
# %% [markdown]
#
# 2. Can we detect intent VPs automatically in task-oriented queries?
#   2.1. How do "intent" VPs differ from non-intent "VPs"?    
#       2.1.1 Candidate hypotheses:
#           - sentence mood: declarative vs. interrogative syntax
#           - tense: present vs. past ?
#           - lexical: some verbs and not others
#           - dependency structure: direct object vs. indirect ?
#
# Observations:  
#   1.1 Grammar features of the VPs that we labelled as intents:
#       - intent -> VB_present + NP
#       - intent -> VB_present + VB_present + NP
#       - intent -> VB_infinitive + NP
#       - intent -> VB_present + clause
#
#   1.2 Grammar features of VPs that we did not label as intents:
#       1.2.1. Implicit, intents at the level of semantics/pragmatics:  
#       - failed intent -> gerund VB | auxiliary VB | past tense VB | interrogative phrase  
#
#       1.2.2. Grammar features of intents that we are not exploiting:
#       - intent -> need | want + VB_infinitive
# %% [markdown]
#### Parse well-formed intent's intent and entities (slot analysis)
# 
# We formalized an intent as follows:
#
#   e.g., : "track my card for me"
#
#   intent -> VB + NP 
#       entity: PP
#       
# POS terminology
#   - SBAR: Subordinate Clause (e.g., after ..)
# %% [markdown]
# CFG FEATURES 
# * What are the signature features of well-formed intent queries?  
#   * structured SVM
#   * clustering in directed graphs (adjacency matrix of pos tags)  
#   * see 2b_eda_VP_graphs.py  
# %%
sorted_annots.head()

# %%
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb

# %% [markdown]  
# # References
#
# https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html    
