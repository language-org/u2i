# author: laquitainesteeve@gmail.com

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction
import stanza
import os
import glob
from src.intent.nodes import setup
import time
from stanza.server import CoreNLPClient

params = setup.get_params()
paths = setup.get_paths()


def instantiate_allennlp_constituency_parser():
    """
    output[trees]
      '(SQ (SBAR (IN If) (S (NP (PRP I)) (VP (VBP bring) (NP (CD 10) (NNS dollars)) (NP (NN tomorrow))))) (, ,) (MD can) (NP (PRP you)) (VP (VB buy) (NP (PRP me)) (NP (NN lunch))) (. ?))'
    - takes: ~0.4 s per sentence (slow)
    """
    return Predictor.from_path(paths["path_constituency_parser"])


def init_Stanza_constituency_parsing(sample):
    """[summary]

    Args:
        sample ([type]): [description]

    Returns:
        [type]: [description]
    """
    tic = time.time()
    with CoreNLPClient(
        annotators=["tokenize", "ssplit", "pos", "lemma", "parse", "depparse", "coref"],
        timeout=30000,
        memory="16G",
        endpoint="http://localhost:8888",
        be_quiet=True,
    ) as client:
        parser = client.annotate(sample)
    print(f"(init_Stanza_constituency_parsing) took {round(time.time()-tic,2)} secs")
    return parser


def setup_stanza():
    """[summary]
    """
    # install corenlp
    if not os.path.exists(paths["coreNLP_path"]):
        print("(setup_stanza) Installing CoreNLP ...")
        stanza.install_corenlp(dir=paths["coreNLP_path"])
    else:
        print("(setup_stanza) CoreNLP already exists.")

    (status, model_file) = is_exist_model()
    if status:
        print("(setup_stanza) CoreNLP model already exists. The model is: {model_file}")
    else:
        print("(setup_stanza) Downloading CoreNLP model ...")
        stanza.download_corenlp_models(
            model=params["corenlp_model"]["lang"],
            version=params["corenlp_model"]["version"],
            dir=paths["coreNLP_path"],
        )

    print("(setup_stanza) Setting up Environment variables ...")
    os.environ["CORENLP_HOME"] = paths["coreNLP_path"]


def is_exist_model():

    model_file = [
        file
        for file in os.listdir(paths["coreNLP_path"])
        if file.endswith(".jar")
        and params["corenlp_model"]["lang"] in file
        and params["corenlp_model"]["version"] in file
    ]
    if len(model_file) == 0:
        return False, model_file
    else:
        return True, model_file

