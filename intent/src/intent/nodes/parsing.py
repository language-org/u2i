# author: laquitainesteeve@gmail.com

import glob
import os
from time import time

import allennlp_models.structured_prediction
import nltk_tgrep
import numpy as np
import pandas as pd
import stanza
from allennlp.predictors.predictor import Predictor
from nltk.tree import ParentedTree
from stanza.server import CoreNLPClient

from intent.src.intent.nodes import parsing, setup

params = setup.get_params()
paths = setup.get_paths()

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
train_data_path = proj_path + "data/01_raw/banking77/train.csv"
test_data_path = proj_path + "data/01_raw/banking77/test.csv"

os.chdir(proj_path)

def init_allen_parser():
    """
    output[trees]
      '(SQ (SBAR (IN If) (S (NP (PRP I)) 
      (VP (VBP bring) (NP (CD 10) (NNS dollars)) 
      (NP (NN tomorrow))))) (, ,) (MD can) 
      (NP (PRP you)) (VP (VB buy) (NP (PRP me)) (NP (NN lunch))) (. ?))'
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
    tic = time()
    with CoreNLPClient(
        annotators=["tokenize", "ssplit", "pos", "lemma", "parse", "depparse", "coref"],
        timeout=30000,
        memory="16G",
        endpoint="http://localhost:8888",
        be_quiet=True,
    ) as client:
        parser = client.annotate(sample)
    print(f"(init_Stanza_constituency_parsing) took {round(time()-tic,2)} secs")
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


def extract_VP(al_prdctor, query):
    """
    Parse and return the first verb phrase of a query

    Args:
        al_prdctor ([type]): [description]
        query ([type]): [description]

    Returns:
        [type]: [description]
    """
    output = al_prdctor.predict(sentence=query)
    parsed_txt = output["trees"]
    tree = ParentedTree.fromstring(parsed_txt)
    verb_p = nltk_tgrep.tgrep_nodes(tree, "VP")
    if not len(verb_p) == 0:
        out = verb_p[0].leaves()
    else:
        out = None
    return out

def extract_all_VPs(prm, data, predictor):
    """[summary]

    Args:
        prm ([type]): [description]
        data ([type]): [description]
        predictor ([type]): [description]

    Returns:
        [type]: [description]
    """

    tic = time()
    n_data = len(data)
    if n_data > prm["sample"]:
        texts = data["text"].sample(prm["sample"])
    else:
        texts= data["text"]
    VPs = []
    dur = []
    for ix in range(len(texts)):
        t0 = time()
        VP = parsing.extract_VP(predictor, texts.iloc[ix])
        if VP is not None:
            VPs.append([' '.join(VP)])
        else:
            VPs.append([])
        if ix <= 10:
            dur.append(time() - t0)
            estTime = round(len(texts) * np.mean(dur), 2)
            print(f'Time to completion: {estTime}')
    print(f'{round(time()-tic,2)}')
    return VPs


def run_parsing_pipe(data:pd.DataFrame, predictor:object, prm:dict, verbose:bool=True) -> pd.DataFrame:
    """[summary]

    Parse the texts contained in rows of a dataframe

    Args:
        prm (dict): [description]
        verbose (bool, optional): [description]. Defaults to True.

    Returns:
        pd.DataFrame: [description]
    """

    t0 = time()

    # select data for an input class
    data_class_i = data[data["category"].isin(prm["intent_class"])]
    sample = data_class_i["text"].iloc[0] # get first VPs, others often variants 
    
    # PARSING
    tic = time()
    output = predictor.predict(sentence=sample)
    parsed_txt = output["trees"]
    if verbose:
        print(f"(run_parsing_pipe)(Inference) took {round(time()-tic,2)} secs")
        print(f"Parsed sample:\n{parsed_txt}\n")

    # VP EXTRACTION
    tree = ParentedTree.fromstring(parsed_txt)
    assert len(parsing.extract_VP(predictor, "I want coffee")) > 0, "VP is Empty\n"

    # Speed up (1 hour / 10K queries)
    VPs = parsing.extract_all_VPs(prm, data_class_i, predictor)
    assert (
        len(VPs) == len(data_class_i) or len(VPs) == prm["sample"]
    ), '''(run_parsing_pipe) VP's length does not match "data_class_i"\n'''

    # add to data, show
    data_class_i["VP"] = pd.DataFrame(VPs)
    assert data_class_i.category.nunique() == len(prm['intent_class']), '''The intent classes in the parsed_data does not match input data's'''
    print(f"(run_parsing_pipe) took {round(time()-tic,2)} secs\n")
    return data_class_i

