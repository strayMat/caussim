"""
Score several g-formula models with mu_iptw_risk (reweighted mse) and mu_risk (mse on y) on different semi-simulated datasets
"""

import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from sklearn.utils import check_random_state
from tqdm import tqdm
from copy import deepcopy
from caussim.data.loading import make_dataset_config
from caussim.experiences.causal_scores_evaluation import run_causal_scores_eval
from caussim.experiences.pipelines import *


from caussim.experiences.base_config import (
    CANDIDATE_FAMILY_HGB_UNDERFIT,
    CANDIDATE_FAMILY_RIDGE_TLEARNERS,
    CATE_CONFIG_ENSEMBLE_NUISANCES,
    CANDIDATE_FAMILY_HGB,
    CATE_CONFIG_LOGISTIC_NUISANCES,
    DATASET_GRID_EXTRAPOLATION_RESIDUALS,
    DATASET_GRID_FULL_EXPES,
    ACIC_2018_PARAMS,
)

from caussim.experiences.utils import compute_w_slurm, set_causal_score_xp_name

RANDOM_STATE = 0
generator = check_random_state(RANDOM_STATE)

SMALL_DATASET_GRID = [
    {
        "dataset_name": ["acic_2016"],
        "dgp": list(range(1, 78)),
        "random_state": list(range(1, 11)),
    },
    {
        "dataset_name": ["caussim"],
        "overlap": generator.uniform(0, 2.5, size=100),
        "random_state": list(range(1, 4)),
        "treatment_ratio": [0.25, 0.5, 0.75],
    },
    {
        "dataset_name": ["twins"],
        "overlap": generator.uniform(0.1, 3, size=100),
        "random_state": list(np.arange(10)),
    },
    {
        "dataset_name": ["acic_2018"],
        "ufid": ACIC_2018_PARAMS.loc[ACIC_2018_PARAMS["size"] <= 5000, "ufid"].values,
    },
]
# DATASET_GRID = DATASET_GRID_FULL_EXPES
CAUSAL_RATIO_GRID = [
    {
        "dataset_name": ["caussim"],
        "overlap": generator.uniform(0, 2, size=4),
        "random_state": list(range(0, 5)),
        "effect_size": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        "treatment_ratio": [0.25],
    },
]  # bigger grid was 25 different overlaps and 5 random states
DATASET_GRID = CAUSAL_RATIO_GRID  # SMALL_DATASET_GRID

# Fixing this parameter to non 0 separate the test set into a train set and a
# test distinct from the nuisance set (kept to the same size)
XP_CATE_CONFIG_SETUP = CATE_CONFIG_ENSEMBLE_NUISANCES.copy()
# XP_CATE_CONFIG_SETUP =  CATE_CONFIG_LOGISTIC_NUISANCES.copy()
XP_CATE_CONFIG_SETUP["separate_train_set_ratio"] = 0

# ### Evaluate several dgps ### #
if __name__ == "__main__":
    t0 = datetime.now()
    parser = argparse.ArgumentParser()
    # parser.add_argument("--xp_name", type=str,default=None,help="xp folder to consolidate",)
    parser.add_argument("--slurm", dest="slurm", default=False, action="store_true")
    parser.add_argument(
        "--extrapolation_plot",
        dest="extrapolation_plot",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--write_to_parquet",
        dest="write_to_parquet",
        default=False,
        action="store_true",
    )
    config, _ = parser.parse_known_args()
    config = vars(config)

    expe_timestamp = datetime.now()
    # Loop on simulations
    # xp_name = config['xp_name']
    simu_grid = []
    for dataset_grid in DATASET_GRID:
        dataset_name = dataset_grid["dataset_name"][0]
        if dataset_name == "caussim":
            candidate_estimators_grid = deepcopy(CANDIDATE_FAMILY_RIDGE_TLEARNERS)
        else:
            candidate_estimators_grid = deepcopy(CANDIDATE_FAMILY_HGB)
        xp_name = set_causal_score_xp_name(
            dataset_name=dataset_name,
            dataset_grid=dataset_grid,
            cate_config=XP_CATE_CONFIG_SETUP,
            candidate_estimators_grid=candidate_estimators_grid,
        )
        for dataset_setup in tqdm(list(ParameterGrid(dataset_grid))):
            dataset_config = make_dataset_config(**dataset_setup)
            cate_config = deepcopy(XP_CATE_CONFIG_SETUP)
            candidate_estimators_grid = deepcopy(candidate_estimators_grid)
            if config["slurm"]:
                simu_grid.append(
                    {
                        "dataset_config": dataset_config,
                        "cate_config": cate_config,
                        "candidate_estimators_grid": candidate_estimators_grid,
                        "xp_name": xp_name,
                        "extrapolation_plot": config["extrapolation_plot"],
                        "write_to_parquet": config["write_to_parquet"],
                    }
                )
            else:
                run_causal_scores_eval(
                    dataset_config=dataset_config,
                    cate_config=cate_config,
                    candidate_estimators_grid=candidate_estimators_grid,
                    xp_name=xp_name,
                    extrapolation_plot=config["extrapolation_plot"],
                    write_to_parquet=config["write_to_parquet"],
                )
    if config["slurm"]:
        compute_w_slurm(run_causal_scores_eval, simu_grid)
    print(f"\n##### Cycle of simulations ends ##### \n Duration: {datetime.now() - t0}")
