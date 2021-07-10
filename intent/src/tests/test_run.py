# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test``.
"""
from collections import defaultdict
from itertools import chain
from pathlib import Path

import pandas as pd
import pytest

from intent.src.intent.nodes import parsing, similarity

# from intent.run import ProjectContext


@pytest.fixture
def project_context():
    return ProjectContext(str(Path.cwd()))


class TestProjectContext:
    def test_project_name(self, project_context):
        assert project_context.project_name == "intent"

    def test_project_version(self, project_context):
        assert project_context.project_version == "0.16.2"


def test_extract_all_VPs(VPs, data, prm):
    assert (
        len(VPs) == len(data) or len(VPs) == prm["sample"]
    ), '''VP's length does not match "data"'''


def test_extract_VP(al_prdctor):
    assert len(parsing.extract_VP(al_prdctor, "I want coffee")) > 0, "VP is Empty"


def test_annots_df(annots_df: pd.DataFrame):
    assert set(annots_df.columns).issuperset(
        {"index", "VP", "annots"}
    ), """ "index", "VP", "annots" columns are missing" """


def simil_matx(simil_matx):
    assert (
        simil_matx.shape[0] + 1 == simil_matx.shape[1]
    ), "similarity matrix shape should be (n, n+1)"
    assert not len(simil_matx) == 0, "similarity matrix is empty"


def test_len_similarity_matx(cfg: pd.DataFrame, sim_matx: pd.DataFrame):
    tag = parsing.from_cfg_to_constituents(cfg["cfg"])

    assert tag.nunique() == len(
        sim_matx
    ), f""" Number of unique constituents in 'cfg' {tag.nunique()} must match 'len(sim_matx)' {len(
        sim_matx)} """


def test_rank_nearest_to_seed(sim_matx: pd.DataFrame, seed: str):
    l_ranked = len(similarity.rank_nearest_to_seed(sim_matx, seed=seed))
    assert l_ranked == len(
        sim_matx
    ), """ The length of 'rank_nearest_to_seed()''s output should match len(sim_matx) """


def test_posting_list(posting_list: dict, sim_matx: pd.DataFrame, seed: str):

    ranked = similarity.rank_nearest_to_seed(sim_matx, seed=seed, verbose=False)
    assert (
        len(set(posting_list.keys()).difference(set(ranked.index))) == 0
    ), """ posting_list and 'rank_nearest_to_seed''s output should have the 
    same set of constituents"""


def test_get_posting_index(
    cfg: pd.DataFrame, posting_list: dict, sorted_series: pd.Series
) -> list:
    l_index = len(similarity.get_posting_index(posting_list, sorted_series))
    assert l_index == len(
        cfg
    ), f""" 'index' length {l_index} must be same as 'cfg' length {len(cfg)} """
