from logging import debug

import pandas as pd
import yaml
from nltk.corpus import wordnet as wn

from . import features

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"

with open(proj_path + "intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)

todf = pd.DataFrame


def sample(tr_data: pd.DataFrame) -> pd.DataFrame:
    """sample dataset rows

    Args:
        tr_data (pd.DataFrame): corpus of queries

    Returns:
        pd.DataFrame: [description]
    """

    # data = tr_data[tr_data["category"].eq(prms["intent_class"])]
    data = tr_data[tr_data["category"].isin(prms["intent_class"])]
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
    return query[count == n_sent]


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
    cfg = pd.concat(
        [cfg_filt, mood_filt],
        ignore_index=False,
        axis=1,
    )
    return cfg


def filter_words_not_in_wordnet(corpus: tuple) -> tuple:
    """Filter mispelled words (absent from wordnet)

    Args:
        corpus (tuple): tuple of queries

    Returns:
        [tuple]: tuple of queries from which mispelled words have been filtered
    """
    # find mispelled words
    misspelled = []
    for query in corpus:
        if query:
            query = query.split()
        for word in query:
            if not wn.synsets(word):
                misspelled.append(word)

    # filter them from corpus
    queries = []
    for query in corpus:
        if query:
            query = query.split()
        filtered = []
        for word in query:
            if not word in misspelled:
                filtered.append(word)
        queries.append(" ".join(filtered))
    return tuple(queries)


def filter_words(corpus: pd.Series, how: str) -> tuple:
    """Filter mispelled words (absent from wordnet)
    [DEPRECATED]

    Args:
        corpus (tuple): tuple of queries
        how (str):
            "not_in_wordnet": remove words not in wordnet
    Returns:
        [tuple]: tuple of queries from which mispelled words have been filtered
    """
    if how == "not_in_wordnet":

        # find mispelled words
        misspelled = []
        for query in corpus:
            if query:
                query = query.split()
            for word in query:
                if not wn.synsets(word):
                    misspelled.append(word)

        # filter them from corpus
        queries = []
        for query in corpus:
            if query:
                query = query.split()
            filtered = []
            for word in query:
                if not word in misspelled:
                    filtered.append(word)
            queries.append(" ".join(filtered))
    return pd.Series(queries, index=corpus.index)


def filter_empty_queries(corpus: tuple) -> tuple:
    """Filter empty queries out of corpus

    Args:
        corpus (tuple): corpus of string queries

    Returns:
        tuple: corpus of string queries without empty queries
    """
    return tuple(filter(None, corpus))


def drop_empty_queries(corpus: pd.Series) -> tuple:
    """Filter empty queries out of corpus

    Args:
        corpus (pd.Series): corpus of string queries

    Returns:
        pd.Series: corpus of string queries without empty queries
    """
    return corpus[~corpus.isin(["", None])]
