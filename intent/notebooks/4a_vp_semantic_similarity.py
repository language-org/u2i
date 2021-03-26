# %%
import os
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from scipy.spatial import distance

from intent.src.intent.nodes import similarity

# %% [markdown]
## DATA
# %%
text = (
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
# distance threshold t - the maximum inter-cluster distance allowed
# %%
DIST_THRES = 1.8
#%%
t0 = time()
sem_sim_matx = similarity.get_semantic_similarity_matrix(text)
print(f"{round(time() - t0, 2)} secs")
sem_sim_matx = pd.DataFrame(sem_sim_matx)
sem_sim_matx

# %%
# patch weird values with -1
sem_sim_matx[np.logical_or(sem_sim_matx < 0, sem_sim_matx > 1)] = -0.1
sem_sim_matx[sem_sim_matx.isnull()] = -1

# %%
row_linkage = hierarchy.linkage(distance.pdist(sem_sim_matx), method="average")
sns.clustermap(
    sem_sim_matx,
    row_linkage=row_linkage,
    method="average",
    figsize=(13, 13),
    cmap="vlag",
)
label = fcluster(row_linkage, t=DIST_THRES, criterion="distance")
pd.DataFrame([text, label]).T
