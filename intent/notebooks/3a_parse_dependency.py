# %% [markdown]
## Visualize dependency tree
#
# author: Steeve Laquitaine
#
# %%
import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
# %%
sample = "track the card you sent me?"
# %%
# parse
doc = nlp(sample)

# %%
# plot dependency tree
displacy.serve(doc, style="dep", minify=True)
