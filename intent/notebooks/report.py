# %%[markdown]
#
# # REPORT
# author: Steeve LAQUITAINE
#
# **EXECUTIVE SUMMARY**
# * A
# * B
#
#
# **APPROACH**
#
# 1. Classification:
#   * _Logistic regression_
#       * Is the lexicon predictive enough to discriminate b/w intents
#           * [Q]: Does low performance impair the quality of our intent parsing ?
#       * Describe the predictive information:
#           * the lexicon clearly differ b/w intents (clustering)
#           * the lexicon is stereotyped & redundant within an intent cluster
#           * [Q]: Do overlapping lexicons or/and diverse lexicons impair the quality of our intent parsing?
#       * Does sentence type impair classification performance ?
#          * [Q] Do Some sentence types have a less discriminative lexicon than others?
# 2. Intent detection:
#   * assess whether an intent is expressed or not
# 3. Intent parsing:
#   * when an intent is expressed: extract it
#       * syntactical parsing?
#       * constituency parsing?
#       * `intent-indicator`
# 4. Compair w/ freer (less stereotyped/constrained) utterances:
#   * Questions expressed in Forums?

# %%
