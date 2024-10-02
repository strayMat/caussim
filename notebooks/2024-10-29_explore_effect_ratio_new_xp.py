# %%
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
import yaml
from sklearn.model_selection import ParameterGrid
from sklearn.utils import check_random_state
from tqdm import tqdm
from caussim.config import DIR2EXPES
from copy import deepcopy

from caussim.reports.utils import get_expe_indices
from caussim.data.loading import load_dataset, make_dataset_config
from caussim.pdistances.effect_size import mean_causal_effect

import numpy as np
# %%
dir2caussim_xp = DIR2EXPES/"caussim_save"
logs_linear = pd.read_parquet(dir2caussim_xp/ "caussim__nuisance_linear__candidates_ridge__overlap_06-224.parquet")
logs_linear.head()
_, linear_xp_indices = get_expe_indices(logs_linear,"effect_ratio")
logs_linear = logs_linear.sort_values(linear_xp_indices+["r_risk"])

logs_non_linear = pd.read_parquet(dir2caussim_xp/"caussim__nuisance_non_linear__candidates_ridge__overlap_06-224.parquet").sort_values(linear_xp_indices+["r_risk"])
logs_non_linear.head()
# %%
logs_linear[linear_xp_indices+["r_risk", "r_risk_gold_e"]].head()
#%%
logs_non_linear[linear_xp_indices+["r_risk", "r_risk_gold_e"]].head()

effect_ratio_distribution = logs_non_linear["effect_ratio"].drop_duplicates().describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
print(effect_ratio_distribution.to_markdown())