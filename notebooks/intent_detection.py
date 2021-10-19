# %% [markdown]
# # INTENT DETECTION
# author: steeve Laquitaine
#
# ### Scope
#
#   * Task-oriented utterances
#   * Banking sector
#
# ### What I learnt
#
#   * Task-oriented utterances are complex: task-oriented utterances are sometimes multi-sentences and multi-sentences types
#       * Let's call this "Utterance complexity" for now.
#       * [Q]: How often?
#
# ### TODO
#   * only filter in declarative & imperative sentences. Currently sentences are mixed.
#
# %%
import pandas as pd

# my intent analysis package
import intent
from intent.nodes.utils import classify_sentence_type, detect_sentence_type
from intent.nodes.utils import UtteranceComplexity


# %%[markdown]
# ### set paths
# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
train_data_path = proj_path + "data/01_raw/banking77/train.csv"
test_data_path = proj_path + "data/01_raw/banking77/test.csv"


# %%
# train and test
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)


# %%
# show
print("Dataset description:\n")
train_data.info()
print("\Stats:\n")
display(train_data.describe())
print("\nPreview:")
train_data.head()


# %%
# focus on declarative & imperative type sentences
#   - my goal is to first remove some variability in the dataset
#   - assumption: they express actionable intents (1)
#   - [TODO]: only filter in declarative & imperative sentences. Currently sentences are mixed.

# classify sentences
sentence_type = classify_sentence_type(train_data.text.tolist())
train_data_feat = train_data.copy()
train_data_feat["sentence_type"] = sentence_type


# %%
# keep only queries made of one sentence
train_data_feat["sentence_count"] = UtteranceComplexity.count_sentences(
    train_data_feat.text.tolist()
)
train_data_feat_filtered = train_data_feat[train_data_feat.sentence_count.eq(1)]
train_data_feat_filtered.head()
print("Number of queries:", len(train_data_feat_filtered))


#%%
# keep declarative sentences
TYPE = "state"
declarative_queries = train_data_feat[
    train_data_feat.apply(lambda x: detect_sentence_type(x.sentence_type, TYPE), axis=1)
]
filtered


# %%
# define an intent
#   - assumption: an intent is actionable: verb -> object


# %%


# %% [markdown]
# # References
#
# (1) Nikhita Vedula et al., “Towards Open Intent Discovery for Conversational Text,” ArXiv:1904.08524 [Cs], April 17, 2019, http://arxiv.org/abs/1904.08524.
