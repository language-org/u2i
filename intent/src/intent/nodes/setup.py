import yaml

proj_path = "/Users/steeve_laquitaine/Desktop/CodeHub/nlp/"
conf_params_path = proj_path + "conf/params.yml"
conf_paths = proj_path + "conf/paths.yml"


def get_params():
    with open(conf_params_path) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    return params


def get_paths():
    with open(conf_paths) as file:
        paths = yaml.load(file, Loader=yaml.FullLoader)
    return paths
