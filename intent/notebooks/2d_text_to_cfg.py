# %% [markdown]
## Convert text to context free grammar
# Author: Steeve Laquitaine
# %% [markdown]
# PACKAGES
# %%
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from intent.src.intent.nodes import parsing, preprocess

with open(proj_path + "intent/conf/base/catalog.yml") as file:
    catalog = yaml.load(file)

# %% [markdown]
# PATHS
# %%
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

raw_data_path = proj_path + "intent/data/01_raw/banking77/train.csv"

# %% [markdown]
# LOAD TEXT DATA
# %%
raw_data = pd.read_csv(raw_data_path)


# %% [markdown]
# PARSE
# %%
# load allen parser
al_prdctor = parsing.init_allen_parser()

# preprocess, parse verb phrases and get production rules
data = preprocess.sample(raw_data)
VP_info = parsing.extract_all_VPs(data, al_prdctor)
VP_info = parsing.get_CFGs(VP_info)
VPs = parsing.make_VPs_readable(VP_info)
data["VP"] = np.asarray(VPs)
data["cfg"] = np.asarray(
    [VP["cfg"] if not len(VP) == 0 else None for VP in VP_info]
)
data.head()
# %%
joblib.dump(sample, catalog["cfg_productions"])
# %%
# text = parsing.from_text_to_cfg(data, al_prdctor)
constt = parsing.from_cfg_to_constituents(text["cfg"])
constt

# %%
productions = joblib.load(catalog["cfg_productions"])

# %%
