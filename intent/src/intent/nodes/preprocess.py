

import pandas as pd
import yaml

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"

with open(proj_path+"intent/conf/base/parameters.yml") as file:
    prms = yaml.load(file)

def sample(tr_data:pd.DataFrame) -> pd.DataFrame:
    """sample dataset rows

    Args:
        tr_data ([type]): [description]

    Returns:
        [type]: [description]
    """
    data = tr_data[tr_data["category"].eq(prms["intent_class"])]
    if len(data) > prms['sampling']["sample"]:
        sample = data.sample(prms['sampling']["sample"], random_state=prms['sampling']['random_state'])
    else:
        sample = data
    # set index as a tracking column
    sample.reset_index(level=0, inplace=True)
    return sample
