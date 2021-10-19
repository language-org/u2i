import os

import pandas as pd

PROJ_PATH = os.getenv("PROJ_PATH")

import logging

import mlflow
import numpy as np
from src.intent.nodes import config, parsing, preprocess, retrieval, similarity
from src.intent.pipelines.parsing import Cfg
from src.intent.pipelines.similarity import Lcs

# prep logging
logger = logging.getLogger()

# load parameters
PARAMS = config.load_parameters(PROJ_PATH)


class Processing:
    """Intent Processing class"""

    def __init__(
        self,
        params: dict,
        num_sent: int,
        filt_mood: tuple,
        intent_score: float,
        seed: str,
        denoising: str,
    ):
        """Instantiate processing class

        Args:
            params ([type]): [description]
            num_sent ([type], optional): [description]. Defaults to None.
            filt_mood ([type], optional): [description]. Defaults to None.
            INTENT_SCORE ([type], optional): [description]. Defaults to None.
            seed ([type], optional): [description]. Defaults to None.
        Returns:
            Instance of Processing class

        """
        self.params = params
        self.NUM_SENT = num_sent
        self.FILT_MOOD = filt_mood
        self.INTENT_SCORE = intent_score
        self.DENOISING = denoising
        self.SEED = seed

        # print and log processing pipeline parameters
        self._print_params()

    def _print_params(self):
        """Print and log pipeline parameters"""

        # print
        logger.info("-------- PROCESSING ------------")
        logger.info("Parameters:\n")
        logger.info(f" Sentences/query: {self.NUM_SENT}")
        logger.info(f" Mood: {self.FILT_MOOD}")
        logger.info(
            f" Threshold similarity score:  {self.INTENT_SCORE}"
        )
        logger.info(f" Seed: {self.SEED}")

        # log
        mlflow.log_param("nb_sentences", self.NUM_SENT)
        mlflow.log_param("mood", self.FILT_MOOD)
        mlflow.log_param("denoising", self.DENOISING)
        mlflow.log_param("seed", self.SEED)
        mlflow.log_param(
            "similarity threshold", self.INTENT_SCORE
        )

    def run(self, corpus) -> pd.DataFrame:
        """Run pipeline

        Args:
            corpus (pd.DataFrame): [description]

        Returns:
            pd.DataFrame: [description]
        """

        # parse constituents
        cfg = Cfg(corpus, self.params).do()
        logger.info("Parsing constituents")
        logger.info(f"N={len(cfg)} queries")

        # drop complex queries
        cfg_cx = preprocess.filter_n_sent_eq(
            cfg, self.NUM_SENT, verbose=True
        )
        logger.info("Filtering complex queries")
        logger.info(f"N={len(cfg_cx)} queries")

        # drop moods
        cfg_mood = preprocess.filter_in_only_mood(
            cfg_cx, self.FILT_MOOD
        )
        logger.info("Filtering moods")
        logger.info(f"N={len(cfg_mood)} queries")

        # get constituents
        tag = parsing.from_cfg_to_constituents(
            cfg_mood["cfg"]
        )

        # filter dissimilar syntax
        similarity_matrix = Lcs().do(cfg_mood)
        sim_ranked = similarity.rank_nearest_to_seed(
            similarity_matrix, seed=self.SEED, verbose=True
        )
        posting_list = retrieval.create_posting_list(tag)
        ranked = similarity.print_ranked_VPs(
            cfg_mood, posting_list, sim_ranked
        )
        filtered = similarity.filter_by_similarity(
            ranked, self.INTENT_SCORE
        )
        logger.info("Filtering non-intent syntax")
        logger.info(f"N={len(filtered)} queries")

        # index queries
        raw_ix = cfg_mood["index"]
        filtered_raw_ix = raw_ix.values[
            filtered.index.values
        ]

        # Infer and parametrize intents
        # 1. Parse queries functionally
        # 2. Extract named entities
        # 3. Build intent (ROOT), intendeed (dobj) and entities (NER)
        intents = parsing.parse_intent(filtered)

        # display (intent, intendeed)
        cfg_mood.index = cfg_mood["index"]
        cfg_mood.merge(
            pd.DataFrame(intents, index=filtered_raw_ix),
            left_index=True,
            right_index=True,
        )[["index", "text", "intent", "intendeed"]]

        # drop words not in wordnet
        wordnet_filtered = preprocess.filter_words(
            cfg_mood["VP"], "not_in_wordnet"
        )

        # drop empty queries
        (
            processed,
            not_empty,
        ) = preprocess.drop_empty_queries(wordnet_filtered)
        loc_kept = np.where(not_empty == True)[0]
        intents = pd.DataFrame(intents).loc[loc_kept]
        return (processed, intents)
