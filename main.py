# Intent inference
# author: Steeve Laquitaine

import os
from time import time

# set environment variables
proj_path = os.getcwd()
os.environ["PROJ_PATH"] = proj_path
os.environ["NLTK_DATA"] = os.path.join(
    proj_path, "data/06_models/nltk_data"
)

import logging
import logging.config

import pandas as pd
import yaml

from src.intent.nodes import config
from src.intent.nodes import evaluation as evaln
from src.intent.nodes.model import U2iModel

# configurate logging
logging_path = os.path.join(
    proj_path + "/conf/base/logging.yml"
)
with open(logging_path, "r") as f:
    LOG_CONF = yaml.load(f, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    """Entry point
    """

    # load parameters
    logger.info("Loading pipeline parameters...")
    prms = config.load_parameters(proj_path)
    DIST_THRES = prms["DIST_THRES"]
    HCL_METHOD = prms["HCL_METHOD"]

    # Loading data
    t0 = time()
    t_read = time()
    corpus_path = (
        proj_path + "/data/01_raw/banking77/train.csv"
    )
    corpus = pd.read_csv(corpus_path)
    logger.info(
        f"Reading dataset took {time()-t_read} secs"
    )

    # train model
    logger.info("Training model...")
    t_train = time()
    model = U2iModel(DIST_THRES, HCL_METHOD, prms)
    fitted, intents = model.fit(corpus)
    logger.info(f"Training took {time()-t_train} secs")

    # infer intent
    logger.info("Calculating preds...")
    t_infer = time()
    pred = model.predict(corpus, fitted)
    logger.info(f"Inference took {time()-t_infer} secs")

    # evaluate model
    logger.info("Evaluating model...")
    metrics = evaln.Metrics(
        ("rand_index", "mutual_info"),
        pred["cluster_labels"],
        pred["true_labels"],
    ).run()
    logger.info(f"Metrics: {metrics}")

    contingency_matrix = evaln.Description(
        ("contingency_table",),
        pred["cluster_labels"],
        pred["true_labels"],
    ).run()
    logger.info(contingency_matrix)
    logger.info(f"Pipeline took {time()-t0} secs")

    print("============== RESULTS ==============")
    print(pred.head())
    print(intents.head())

    # clean up caches
    os.system("rm -f ~/.allenlp")
