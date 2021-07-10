# %% [markdown]
# Evaluate clusters
# %% [markdown]
# author: steeve laquitaine
# %% [markdown]
# * approach:
#   * we borrow from decision tree partitioning logic and consider clusters as partitions at terminal nodes
#   * each partition is associated with a true label distribution
#   * each partition label is the distribution's mode and label confidence can be
#       * conditional probability of label given a partition instance, p(label/partition)
#       * partition true label distribution's variance
# %%
