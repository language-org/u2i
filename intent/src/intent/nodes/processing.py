from mlflow.tracking.client import MlflowClient
import pandas as pd
import os

proj_path = "/Users/steeve_laquitaine/desktop/CodeHub/intent/"
os.chdir(proj_path)

# custom packages
from intent.src.intent.pipelines.parsing import Cfg
from intent.src.intent.nodes import config, preprocess, similarity, parsing, retrieval
from intent.src.intent.pipelines.similarity import Lcs  # [move to node]

# experiment tracking
import mlflow
from urllib.parse import urlparse

# logging
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# load parameters
PARAMS = config.load_parameters(proj_path)


class Processing:
    """Intent Processing class"""

    def __init__(
        self,
        params: dict,
        num_sent: int,
        filt_mood: tuple,
        thres_sim_score: float,
        seed: str,
        denoising: str,
    ):
        """Instantiate processing class

        Args:
            params ([type]): [description]
            num_sent ([type], optional): [description]. Defaults to None.
            filt_mood ([type], optional): [description]. Defaults to None.
            thres_sim_score ([type], optional): [description]. Defaults to None.
            seed ([type], optional): [description]. Defaults to None.
        Returns:
            Instance of Processing class

        """
        self.params = params
        self.NUM_SENT = num_sent
        self.FILT_MOOD = filt_mood
        self.THRES_SIM_SCORE = thres_sim_score
        self.DENOISING = denoising
        self.SEED = seed

        # print and log processing pipeline parameters
        self._print_params()

    def _print_params(self):
        """Print and log processing pipeline parameters"""

        # print
        print("\n(Processing) Processing parameters\n")
        print("(Processing) Number of sentences per query: ", self.NUM_SENT)
        print("(Processing) Mood: ", self.FILT_MOOD)
        print("(Processing) Threshold similarity score: ", self.THRES_SIM_SCORE)
        print("(Processing) Seed: ", self.SEED)

        # log
        mlflow.log_param("nb_sentences", self.NUM_SENT)
        mlflow.log_param("mood", self.FILT_MOOD)
        mlflow.log_param("denoising", self.DENOISING)
        mlflow.log_param("seed", self.SEED)
        mlflow.log_param("similarity threshold", self.THRES_SIM_SCORE)

    def run(self, corpus) -> pd.DataFrame:
        """Run instantiated processing pipeline

        Args:
            corpus (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """
        # constituency parsing
        cfg = Cfg(corpus, self.params).do()

        # filter queries by complexity
        cfg_cx = preprocess.filter_n_sent_eq(cfg, self.NUM_SENT, verbose=True)

        # filter queries by mood
        cfg_mood = preprocess.filter_in_only_mood(cfg_cx, self.FILT_MOOD)

        # extract constituents
        tag = parsing.from_cfg_to_constituents(cfg_mood["cfg"])

        # filter syntax similarity
        # calculate similarity
        similarity_matrix = Lcs().do(cfg_mood)
        sim_ranked = similarity.rank_nearest_to_seed(
            similarity_matrix, seed=self.SEED, verbose=True
        )
        posting_list = retrieval.create_posting_list(tag)
        ranked = similarity.print_ranked_VPs(cfg_mood, posting_list, sim_ranked)
        filtered = similarity.filter_by_similarity(ranked, self.THRES_SIM_SCORE)

        # map the queries with their raw indices
        raw_ix = cfg_mood["index"]
        filtered_raw_ix = raw_ix.values[filtered.index.values]

        # Intent parsing
        # 1. Apply dependency parsing to each query
        # 2. Apply NER
        # 3. Retrieve (intent (ROOT), intendeed (dobj), entities (NER))
        intents = parsing.parse_intent(filtered)

        # show (intent, intendeed)
        cfg_mood.index = cfg_mood["index"]
        cfg_mood.merge(
            pd.DataFrame(intents, index=filtered_raw_ix),
            left_index=True,
            right_index=True,
        )[["index", "text", "intent", "intendeed"]]

        # filter words not in wordnet
        word_filtered = preprocess.filter_words(cfg_mood["VP"], "not_in_wordnet")

        # drop empty queries
        empty_query_dropped = preprocess.drop_empty_queries(word_filtered)

        return empty_query_dropped
