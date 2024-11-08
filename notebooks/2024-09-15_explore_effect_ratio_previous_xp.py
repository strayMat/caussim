# %%
import pandas as pd
import matplotlib.pyplot as plt

from caussim.data.causal_df import CausalDf, mean_causal_effect_symetric
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
RANDOM_STATE = 0

generator = check_random_state(RANDOM_STATE)
CAUSAL_RATIO_GRID = [
    {
        "dataset_name": ["caussim"],
        "overlap": generator.uniform(0, 2, size=4),
        "random_state": list(range(0, 5)),
        "effect_size": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        "treatment_ratio": [0.25],
    },
]  # bigger grid was 25 different overlaps and 5 random states
# %%
xp_name = "caussim__nuisance_non_linear__candidates_ridge__overlap_35-139.parquet"
caussim_logs = pd.read_parquet(DIR2EXPES/f"caussim_save/{xp_name}")
caussim_logs.head()

# %%
overlap_param, expe_indices = get_expe_indices(caussim_logs)

unique_indices = caussim_logs[expe_indices].drop_duplicates()
print(unique_indices)

# %% 
# I need to rerun the exact same simulations and make the join on the overlap parameter and test seed.  
delta_mu_distrib = []
for dataset_setup in tqdm(list(ParameterGrid(CAUSAL_RATIO_GRID))[:5]):
    dataset_config = make_dataset_config(**dataset_setup)
    sim, dgp_sample = load_dataset(dataset_config=dataset_config)
    sim.rs_gaussian = dataset_config["test_seed"] + 1
    df_nuisance_set = sim.sample(num_samples=dataset_config["train_size"]).df
    sim.rs_gaussian = dataset_config["test_seed"]
    df_test = sim.sample(num_samples=dataset_config["test_size"]).df
    delta_mu_s = mean_causal_effect_symetric(dgp_sample.df.mu_1, dgp_sample.df.mu_0)
    description_test = CausalDf(df_test.reset_index(drop=True)).describe(prefix="test_")
    delta_mu_distrib.append(delta_mu_s)
# %%
print(pd.DataFrame(delta_mu_distrib).describe(
    percentiles=np.array([1, 10, 25, 50, 60, 65, 70, 75, 90, 99])/100).T.to_markdown(index=False)
    )
plt.hist(delta_mu_distrib, bins=100)
plt.xlim(0, 50)