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
#
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

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)
# in root
from intent.src.intent.nodes import mood, parsing

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
tr_data_path = proj_path + "intent/data/01_raw/banking77/train.csv"
test_data_path = proj_path + "intent/data/01_raw/banking77/test.csv"
# %% [markdown]
## PARAMETERS
prm = dict()
prm["sample"] = 100
prm["mood"] = ["declarative"]
prm["intent_class"] = "card_arrival"  # good parsing performance
# %%
# read queries data
tr_data = pd.read_csv(tr_data_path)
# %%
# select data for an input class
data = tr_data[tr_data["category"].eq(prm["intent_class"])]
# %%
data.head(5)
# %%
sample = data["text"].iloc[0]
# %% [markdown]
## PARSING
# %% [markdown]
### ALLENLP VP PARSING
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
# %%
tree = ParentedTree.fromstring(parsed_txt)
assert len(parsing.extract_VP(al_prdctor, "I want coffee")) > 0, "VP is Empty"
# %%
# Speed up (1 hour / 10K queries)
VPs = parsing.extract_all_VPs(prm, data, al_prdctor)
assert (
    len(VPs) == len(data) or len(VPs) == prm["sample"]
), '''VP's length does not match "data"'''
# %%
# augment dataset with VPs
data["VP"] = pd.DataFrame(VPs)
data.iloc[: prm["sample"]]
# %% [markdown]
# Write parsed data
# %%
data.to_excel(catalog['parsed'])
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
# 1. Annotate VPs that look like intent vs. not
# 2. Look what make them different
# 3. Test a few hypothesis:
#   - mood: declarative vs. interrogative syntax ?
#   - tense: present vs. past ?
#   - lexical: some verbs and not others
#   - else ?
#   - semantics: direct object vs. indirect ?
# %%
annots = annotate(data["VP"], options=["yes", "no"])
# %% [markdown]
# Write annots
# %%
if not os.path.isfile(catalog['annots']):
    # add current time to filename 
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S").replace(' ','_').replace(':','_').replace('/','_')    
    filepath = os.path.splitext(catalog['annots'])
    myfile, myext = filepath[0], filepath[1]
    pd.DataFrame(annots).to_excel(f'{myfile}_{now}{myext}')
else:
    print('WARNING: Annots was not written. To write, delete existing and rerun.')
# %%
annots_df = pd.DataFrame(annots, columns=['text','annot'])
annots_df['annot'][annots_df['text'].isnull()] = np.nan
# %% [markdown]
# **Fig. Queries are sorted by annot class below.**
# %%
annots_df = annots_df.sort_values(by='annot', ascending=False)
# %% [markdown]
# to convert to pdf  
# %%
annots_df
# %% [markdown]
# **Fig. Proportion of yes.**
# %%
n_total = len(annots_df)
n_null = annots_df['annot'].isnull().sum()
n_yes = annots_df['annot'].eq('yes').sum()
n_no = annots_df['annot'].eq('no').sum()
stats = pd.DataFrame({
    'annot': ['null', 'yes', 'no','Total'], 
    'count': [n_yes, n_no, n_null, n_total],
    '%': [n_yes/n_total, n_no/n_total, n_null/n_total, 1]
    })
stats
# %%
# jupyter nbconvert --no-input --to=pdf 2_Intent_parsing.ipynb

