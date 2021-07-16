# author: steeve laquitaine

import yaml


def load_parameters(proj_path):
    """load parameters from conf/parameters.yml

    Args:
        proj_path (str): [description]

    Returns:
        dict: [description]
    """
    with open(proj_path + "intent/conf/base/parameters.yml") as file:
        prms = yaml.load(file)
    return prms
