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
#          * [Q] Does performance change w/ sentence sub-type (closed vs open-form question...see p392, Jurafsky)
#       * Can we detect implicit vs. explicit intent and assess their impact on performance?
# 2. Intent detection:
#   * assess whether an intent is expressed or not
# 3. Simple Intent parsing:
#   * when an intent is expressed: extract it
#       * syntactical parsing?
#       * constituency parsing?
#       * `intent-indicator`
# 4. Compare w/ less stereotyped utterances:
#   * [Q] Does our simple intent parsing generalize to `freer` dataset?
#   * Crowd-sourced:
#       * Forums
#           * `Clinc`: 23,700 instances, 150 intents, cross-domain
#       * Twitter
#   * `Liu 2019`: 25,716 instances, 64 intents, cross-domain, task-oriented
# 5. Is there more predictive information e `semantic graph` for each intent
# than in lexicon?
# %%
