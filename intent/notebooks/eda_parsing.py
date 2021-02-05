# %% [markdown]
## Explore VP parsing results
# author: steeve laquitaine  
# * Summary:    
#   * Take a sample of N queries from each intent class and parse VPs
#   * calculate parsing performances:  
#       * parsed / or not
#       * intent / or not
# %%
import os
from time import time

import pandas as pd

from intent.src.intent.nodes import parsing

to_df = pd.DataFrame
# %% [markdown]
## PARAMETERS
prm = dict()
prm['sample'] = 100
prm['mood'] = ['declarative']
prm[
    'intent_class'
] = ['card_arrival', 'card_linking', 'exchange_rate',
       'card_payment_wrong_exchange_rate', 'extra_charge_on_statement',
       'pending_cash_withdrawal', 'fiat_currency_support',
       'card_delivery_estimate', 'automatic_top_up', 'card_not_working',
       'exchange_via_app'] # take a sample of 10 classes
# %%
# set data path
proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
data_path = proj_path + "data/01_raw/banking77/train.csv"
os.chdir(proj_path)
# %%
# read data
data = pd.read_csv(data_path)
# %% [markdown]
## PARSING PERFORMANCE
# %%
# instantiate parser
tic = time()
predictor = parsing.init_allen_parser()
print(f"(run_parsing_pipe)(Instantiation) took {round(time()-tic,2)} secs")
# %%
# parse (40 sec / 100 samples =~ 1h for dataset)
parsed_data = parsing.run_parsing_pipe(data, predictor, prm, verbose=True)
parsed_data
# %% [markdown]
## PARSING PERFORMANCE
# Parsed / or not
# %% [markdown]
# **Fig**. Performance for across intent classes
# %%
def get_perf(parsed_data):
    return round(100 - parsed_data['VP'].isnull().sum()/len(parsed_data['VP'])*100,2)
perf_parse_or_not = get_perf(parsed_data)
print(f'The "parse/or not" performances are : {perf_parse_or_not} %')
# %% [markdown]
# **Fig**. Performance per intent class
# * Only `card_arrival` had non-zero performance (61%)
# * All other intent classes had 0% performance (they always failed to parsed VPs)
# %%
df = parsed_data.groupby(by=['category']).apply(lambda x: get_perf(x))
to_df(df, columns=['Performance (%)'])
# %%
print('Done')
