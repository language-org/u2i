# %% [markdown]
## INTENT PARSING
# %% [markdown]
# * **Purpose** :
#   * Test intent parsing with ALLENLP
# %%[markdown]
## SETUP
# %%
from time import time
import pandas as pd

from src.intent.nodes import parsing
from nltk.tree import ParentedTree
import nltk_tgrep

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
## PARSING
# %% [markdown]
### ALLENLP
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
### VP EXTRACTION
# %%
tree = ParentedTree.fromstring(all_parsed_sample)
# %%
verb_p = nltk_tgrep.tgrep_nodes(tree, "VP")
verb_p[0].pretty_print()
verb_p[0].leaves()

