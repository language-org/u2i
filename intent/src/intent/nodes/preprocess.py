import pandas as pd
import yaml
from numpy.lib.twodim_base import triu_indices

from . import features

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"

with open(proj_path + "intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)

todf = pd.DataFrame


def sample(tr_data: pd.DataFrame) -> pd.DataFrame:
    """sample dataset rows

    Args:
        tr_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    data = tr_data[tr_data["category"].eq(prms["intent_class"])]
    if len(data) > prms["sampling"]["sample"]:
        sample = data.sample(
            prms["sampling"]["sample"],
            random_state=prms["sampling"]["random_state"],
        )
    else:
        sample = data

    # set index as a tracking column
    sample.reset_index(level=0, inplace=True)
    return sample


def filter_by_sent_count(
    query: pd.DataFrame, thresh: int, verbose: bool = False
) -> pd.Series:
    """Filter queries dataset, keep query rows w/ N <= thresh sentences

    Args:
        query (pd.DataFrame): dataset of queries in column 'text' each made 
            of 1 to N sentences
        threshold (int): max number of sentences per query allowed in filtered dataset  
        verbose: (bool)

    Returns:
        pd.Series: filtered queries
    """
    count = pd.Series(features.count(query["text"]))
    if verbose:
        print(f"There are {len(count)} original queries.")
        print(
            f"{len(query[count <= thresh])} after filtering < {thresh} sentence queries."
        )
    return query[count <= thresh]


def filter_n_sent_eq(
    query: pd.DataFrame, n_sent: int, verbose: bool = False
) -> pd.Series:
    """Filter queries dataset, keep query rows w/ N = n_sent sentences

    Args:
        query (pd.DataFrame): dataset of queries in column 'text' each made 
            of 1 to N sentences
        threshold (int): number of sentences per query allowed in filtered dataset  
        verbose: (bool)

    Returns:
        pd.Series: filtered queries
    """
    count = pd.Series(features.count(query["text"]))
    if verbose:
        print(f"There are {len(count)} original queries.")
        print(
            f"{len(query[count == n_sent])} after filtering = {n_sent} sentence queries."
        )
    return query[count <= n_sent]


def filter_in_only_mood(cfg: pd.DataFrame, FILT_MOOD: str) -> pd.Series:

    mood_set = ("ask", "state", "wish-or-excl")
    to_drop = set(mood_set).difference(set(FILT_MOOD))
    query_moods = features.classify_sentence_type(cfg["text"])

    # filter indices
    ix = [
        ix
        for ix, mood in enumerate(query_moods)
        if not set(mood).isdisjoint(set(FILT_MOOD))
        and set(mood).isdisjoint(set(to_drop))
    ]

    # add moods to data
    mood_filt = todf(query_moods).iloc[ix].reset_index()
    mood_filt.columns = [f"mood_{i}" for i in range(len(mood_filt.columns))]

    cfg_filt = cfg.iloc[ix].reset_index()
    cfg = pd.concat([cfg_filt, mood_filt], ignore_index=False, axis=1,)
    return cfg
