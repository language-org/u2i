import os

import pandas as pd

PROJ_PATH = os.getenv("PROJ_PATH")

import logging
from time import time
import mlflow
import numpy as np
from src.intent.nodes import (
    config,
    parsing,
    preprocess,
    retrieval,
    similarity,
)
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
        inspect: bool,
    ):
        """Instantiate processing class

        Args:
            params (dict): [description]
            num_sent (int, optional): [description]. Defaults to None.
            filt_mood (str, optional): [description]. Defaults to None.
            INTENT_SCORE (float, optional): [description]. Defaults to None.
            seed (str, optional): [description]. Defaults to None.
            inspect (bool, optional): [description]. Defaults to None.
        Returns:
            Instance of Processing class

        """
        self.params = params
        self.NUM_SENT = num_sent
        self.FILT_MOOD = filt_mood
        self.INTENT_SCORE = intent_score
        self.DENOISING = denoising
        self.SEED = seed
        self.inspect = inspect

        # print and log processing pipeline parameters
        self._print_params()

    def _print_params(self):
        """Print and log pipeline parameters"""

        # print
        logger.info("-------- PROCESSING ------------")
        logger.info("Parameters:")
        logger.info(f"- Sentences/query: {self.NUM_SENT}")
        logger.info(f"- Mood: {self.FILT_MOOD}")
        logger.info(
            f" Threshold similarity score:  {self.INTENT_SCORE}"
        )
        logger.info(f"- Seed: {self.SEED}")

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
        # tape queries
        introsp = corpus["text"].to_frame()

        # parse structure (bottleneck)
        data = self.parse_struct(corpus, introsp)

        # choose complexity
        data = self.filter_cplx(data)

        # choose mood
        data = self.filter_mood(data)

        # get constituents
        (
            introsp,
            similarity_matrix,
            intents_df,
        ) = self.parse_intent_and_slots(data)

        # display (intent, intendeed)
        cfg_mood.index = cfg_mood["index"]
        cfg_mood = cfg_mood.merge(
            intents_df, left_index=True, right_index=True,
        )[
            [
                "index",
                "VP",
                "text",
                "intent",
                "intendeed",
                "mood_1",
            ]
        ]

        # drop words not in wordnet
        wordnet_filtered = preprocess.filter_words(
            cfg_mood["VP"], "not_in_wordnet"
        )

        # drop empty queries
        (
            processed,
            not_empty,
        ) = preprocess.drop_empty_queries(wordnet_filtered)
        raw_ix = wordnet_filtered.index[not_empty]
        intents = intents_df.loc[raw_ix]

        # write representations
        # rep flow
        self.write_internal_rep(introsp)

        # syntax similarity
        self.write_syntx_sim(similarity_matrix)
        return (processed, intents)

    def parse_intent_and_slots(self, data):
        (
            similarity_matrix,
            filtered,
            filtered_raw_ix,
        ) = self.filter_syntax(data)

        # Inference & slot filling
        intents = parsing.parse_intent(filtered)
        intents_df = pd.DataFrame(
            intents, index=filtered_raw_ix
        )

        # inspect
        if self.inspect:
            introsp = self._inspect_intent(
                introsp, intents_df
            )

        return introsp, similarity_matrix, intents_df

    def filter_syntax(self, data):
        tag = parsing.chunk_cfg(data["data"]["cfg"])

        # filter syntax
        t_sx = time()
        similarity_matrix, filtered = self._filter_syntax(
            data["data"], tag
        )
        self._log_syntax(t_sx, filtered)

        # inspect
        if self.inspect:
            introsp = self._inspect_syntax(
                introsp, filtered
            )

        # index
        raw_ix = data["data"]["index"]
        filtered_raw_ix = raw_ix.values[
            filtered.index.values
        ]

        return similarity_matrix, filtered, filtered_raw_ix

    def filter_mood(self, data):
        """Choose mood

        Args:
            data ([type]): [description]

        Returns:
            [type]: [description]
        """
        t_mood = time()
        introsp = data["introsp"]
        data = data["data"]
        cfg_mood = preprocess.filter_mood(
            data, self.FILT_MOOD
        )
        self._log_mood(t_mood, cfg_mood)

        # inspect
        if self.inspect:
            introsp["good_mood"] = None
            kept = introsp.iloc[:, -2].notnull()
            introsp["good_mood"].loc[kept] = 0
            introsp["good_mood"].loc[cfg_mood["index"]] = 1
        return {"data": cfg_mood, "introsp": introsp}

    def write_syntx_sim(self, similarity_matrix):
        file_path = os.path.join(
            PROJ_PATH,
            "data/08_introspection/syntax_similarity.csv",
        )
        similarity_matrix.to_csv(file_path)

    def write_internal_rep(self, introsp):
        file_path = os.path.join(
            PROJ_PATH,
            "data/08_introspection/representations.csv",
        )
        introsp.to_csv(file_path)

    def filter_cplx(self, data):
        """Choose level of complexity

        Args:
            data (Dict[str, Any]): 
                "data":
                    data 
                "introsp":
                    taped internal representations

        Returns:
            Dict[str, Any]: [description]
                "data": 
                    filtered data
                "introsp" (pd.DataFrame):
                    taped internal representations
        """
        t_cx = time()
        introsp = data["introsp"]
        data = data["data"]
        cfg_cx = preprocess.filter_n_sent_eq(
            data, self.NUM_SENT, verbose=True
        )
        self._log_cpx(t_cx, cfg_cx)

        # inspect
        if self.inspect:
            introsp = self._inspect_cplx(
                introsp, data, cfg_cx
            )
        return {"data": data, "introsp": introsp}

    def parse_struct(self, corpus, introsp):
        """parse queries structure

        Args:
            corpus ([type]): [description]
            introsp ([type]): [description]

        Returns:
            [type]: [description]
        """
        t_cfg = time()
        cfg = Cfg(corpus, self.params).do()
        self._log_cfg(t_cfg, cfg)

        # inspect
        if self.inspect:
            introsp = self._inspect_cfg(introsp, cfg)
        return {"data": cfg, "introsp": introsp}

    def _filter_syntax(self, cfg_mood, tag):
        """Process syntax

        Args:
            cfg_mood ([type]): [description]
            tag ([type]): [description]

        Returns:
            [type]: [description]
        """
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

        return similarity_matrix, filtered

    def _log_syntax(self, t_sx, filtered):
        """Log syntax processing

        Args:
            t_sx ([type]): [description]
            filtered ([type]): [description]
        """
        logger.info("Filtering 'not-intent' syntax")
        logger.info(f"N={len(filtered)} queries left")
        logger.info(f"took {time()-t_sx} secs")

    def _log_mood(self, t_mood, cfg_mood):
        """Log mood processing

        Args:
            t_mood ([type]): [description]
            cfg_mood ([type]): [description]
        """
        logger.info("Filtering moods")
        logger.info(f"N={len(cfg_mood)} queries left")
        logger.info(f"took {time()-t_mood} secs")

    def _log_cpx(self, t_cx, cfg_cx):
        """Log complexity processing

        Args:
            t_cx ([type]): [description]
            cfg_cx ([type]): [description]
        """
        logger.info("Filtering complex queries")
        logger.info(f"N={len(cfg_cx)} queries left")
        logger.info(f"took {time()-t_cx} secs")

    def _log_cfg(self, t_cfg, cfg):
        """Log context free grammar parsing

        Args:
            t_cfg ([type]): [description]
            cfg ([type]): [description]
        """
        logger.info("Parsing constituents")
        logger.info(f"N={len(cfg)} queries left")
        logger.info(f"took {time()-t_cfg} secs")

    def _inspect_intent(self, introsp, intents_df):
        """Inspect processed intent

        Args:
            introsp ([type]): [description]
            intents_df ([type]): [description]

        Returns:
            [type]: [description]
        """

        kept = introsp["good_intent_syntx"].notnull()
        for col in intents_df.columns:
            introsp[col] = None
            introsp[col].loc[kept] = 0
            introsp[col].loc[intents_df.index] = intents_df[
                col
            ]
        return introsp

    def _inspect_syntax(self, introsp, filtered):
        """Inspect processed syntax 

        Args:
            introsp ([type]): [description]
            filtered ([type]): [description]

        Returns:
            [type]: [description]
        """
        introsp["good_intent_syntx"] = None
        kept = introsp["good_cplx"].notnull()
        introsp["good_intent_syntx"].loc[kept] = 0
        introsp["good_intent_syntx"].loc[filtered.index] = 1
        return introsp

    def _inspect_cplx(self, introsp, cfg, cfg_cx):
        """Inspect processed complexity

        Args:
            introsp ([type]): [description]
            cfg ([type]): [description]
            cfg_cx ([type]): [description]

        Returns:
            [type]: [description]
        """
        introsp["good_cplx"] = None
        introsp["good_cplx"].loc[cfg["index"]] = 0
        introsp["good_cplx"].loc[cfg_cx["index"]] = 1
        return introsp

    def _inspect_cfg(self, introsp, cfg):
        """Inspect processed context free grammar

        Args:
            introsp ([type]): [description]
            cfg ([type]): [description]

        Returns:
            [type]: [description]
        """
        introsp["VP"] = None
        introsp["cfg"] = None
        introsp["VP"].loc[cfg["index"]] = cfg["VP"]
        introsp["cfg"].loc[cfg["index"]] = cfg["cfg"]
        return introsp
