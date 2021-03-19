from collections import defaultdict

import pandas as pd


def create_posting_list(simil_matx: pd.DataFrame) -> dict:
    """Create a posting list with key:value made of constituents:index in dataframe

    Args:
        simil_matx (pd.DataFrame): [description]

    Returns:
        dict: posting list (a dictionary of listed position indices)
    """
    posting_list = defaultdict(list)
    for ix, cfg in enumerate(simil_matx["cfg"]):
        posting_list[cfg].append(ix)
    return posting_list

