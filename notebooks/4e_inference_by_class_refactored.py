# Intent inference
# author: Steeve Laquitaine
import os
from time import time

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/intent/"
proj_path = os.getcwd()

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
    config = yaml.load(f, Loader=yaml.FullLoader)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)
logger.info("Contest is starting")

# load parameters
prms = config.load_parameters(proj_path)
DIST_THRES = prms["DIST_THRES"]
HCL_METHOD = prms["HCL_METHOD"]

# Loading data
t0 = time()
t_read = time()
corpus_path = proj_path + "/data/01_raw/banking77/train.csv"
corpus = pd.read_csv(corpus_path)
logging.info(f"Reading dataset took {time()-t_read} secs")

# train model
logging.info("Training model")
t_train = time()
model = U2iModel(DIST_THRES, HCL_METHOD, prms)
fitted = model.fit(corpus)
logging.info(f"Training model took {time()-t_train} secs")

# infer intent
logging.info("Calculating predictions..")
t_infer = time()
prediction = model.predict(corpus, fitted)
logging.info(f"Inference model took {time()-t_infer} secs")

# evaluate model
logging.info("Evaluating performances..")
metrics = evaln.Metrics(
    ("rand_index", "mutual_info"),
    prediction["cluster_labels"],
    prediction["true_labels"],
).run()
logging.info(metrics)

contingency_matrix = evaln.Description(
    ("contingency_table",),
    prediction["cluster_labels"],
    prediction["true_labels"],
).run()
logging.info(contingency_matrix)

logging.info(f"Pipeline took {time()-t0} secs")

