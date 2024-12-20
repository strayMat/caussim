import logging
from typing import Dict, List, Tuple
import re

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess
from moepy import lowess as moepy_lowess

from matplotlib import cm
from matplotlib.lines import Line2D

from scipy.stats import mannwhitneyu

from caussim.config import (
    COLOR_MAPPING,
    DIR2FIGURES,
    DIR2NOTES,
    LABEL_MAPPING,
    TAB_COLORS,
)
from caussim.reports.utils import (
    kendalltau_stats,
    get_rankings_aggregate,
    get_metric_rankings_by_dataset,
    get_expe_indices,
    get_candidate_params,
)

"""
Plot utils for reports
"""

CAUSAL_METRICS = [
    "mu_risk",
    "mu_iptw_risk",
    "oracle_mu_iptw_risk",
    "w_risk",
    "oracle_w_risk",
    "u_risk",
    "oracle_u_risk",
    # "mu_ipw_iht_risk",
    # "mu_ipw_n_risk",
    # "oracle_r_risk_ipw",
    # "oracle_r_risk_IS2",
    "r_risk",
    "oracle_r_risk",
    # "r_risk_gold_e",
    # "r_risk_gold_m",
    # "r_risk_ipw",
    # "r_risk_IS2",
    "tau_risk",
]
METRIC_TYPE = "Type of metric"
SEMI_ORACLE_LABEL = "Semi-oracle"
FEASIBLE_LABEL = "Feasible"
ORACLE_PALETTE = {
    SEMI_ORACLE_LABEL: "lightgrey",
    FEASIBLE_LABEL: "dimgray",
}

CAUSAL_METRIC_LABELS = {
    "mu_risk": r"$\widehat{\mu\mathrm{-risk}}$",
    "oracle_mu_iptw_risk": r"$\widehat{\mu\mathrm{-risk}}^{*}_{IPW}$",
    "mu_iptw_risk": r"$\widehat{\mu\mathrm{-risk}}_{IPW}$",
    # "mu_itw_iht_risk": r"$\widehat{\mu\mathrm{-risk}}_{IPW-IHT}$",
    "mu_ipw_n_risk": r"$\widehat{\mu\mathrm{-risk}}_{IPW-N}$",
    "oracle_r_risk": r"$\widehat{\mathrm{R-risk}}^{*}$",
    "oracle_r_risk_IS2": r"$\widehat{\mathrm{R-risk}}_{IS2}^{*}$",
    "r_risk": r"$\widehat{\mathrm{R-risk}}$",
    "r_risk_gold_e": r"$\widehat{\mathrm{R-risk}}_{e^*, m}$",
    "r_risk_gold_m": r"$\widehat{\mathrm{R-risk}}_{e, m^*}$",
    "oracle_r_risk_ipw": r"$\widehat{\mathrm{R-risk}}^{*}_{IPW}$",
    "r_risk_ipw": r"$\widehat{\mathrm{R-risk}}_{IPW}$",
    "r_risk_IS2": r"$\widehat{\mathrm{R-risk}}_{IS2}$",
    "mse_ate": r"$(\tau - \hat \tau)^2$",
    "tau_risk": r"$\widehat{\tau\mathrm{-risk}}$",
    "oracle_upper_bound": r"$2\widehat{\mu\mathrm{-risk}}_{IPW^*} - \widehat{tau\mathrm{-risk}} - 4 \sigma_{Bayes}^2$",
    "upper_bound": r"$2\widehat{\mu\mathrm{-risk}}_{IPW^*} - \widehat{tau\mathrm{-risk}}$",
    "oracle_u_risk": r"$\widehat{\mathrm{U-risk}}^{*}$",
    "u_risk": r"$\widehat{\mathrm{U-risk}}$",
    "oracle_w_risk": r"$\widehat{\tau\mathrm{-risk}}_{IPW}^{*}$",
    "w_risk": r"$\widehat{\tau\mathrm{-risk}}_{IPW}$",
}
EVALUATION_METRIC_LABELS = {
    "normalized_abs_bias_ate": r"$|\frac{\tau - \tau_{\hat f^*_{\mathcal{M}}}}{\tau}|$",
    "normalized_bias_tau_risk_to_best_method": r"""
        $log \frac{\tau \mathrm{Risk}(f^*_{\tau \mathrm{Risk}}) - \tau \mathrm{Risk}(f^*_{\ell})}{\tau \mathrm{Risk}(f^*_{\tau \mathrm{Risk}})}$""",
    "tau_risk": r"$\widehat{\tau Risk}(f^*_{\ell})$",
    "bias_tau_risk_to_best_method": r"""$log \tau \mathrm{Risk}(f^*_{\tau \mathrm{Risk}}) - \tau \mathrm{Risk}(f^*_{\ell})$""",
    "kendalltau_stats": r"""Kendall rank correlation 
    $\kappa (\ell, \tau\mathrm{-Risk})$""",
    "kendalltau_stats_r_risk_only": r"Kendall rank correlation $\kappa (R\mathrm{-Risk}), \tau\mathrm{-Risk})$",
    "kendalltau_stats__ref_oracle_r_risk": r"$\kappa (\ell,\tau\mathrm{-Risk}) - \kappa(\widehat{R\mathrm{-risk}}^*, \tau\mathrm{-Risk})$",
    "kendalltau_stats__ref_oracle_r_risk_long": r"$\kappa (\ell, \tau\mathrm{-Risk}) - \kappa(\widehat{R\mathrm{-risk}}^*, \tau\mathrm{-Risk})$"
    "",
    "kendalltau_stats__r_risk__ref_oracle_r_risk_long": r"""Relative Kendall to semi-oracle $\widehat{R\mathrm{-risk}}^*$""",  # $\kappa (\widehat{R\mathrm{-risk}}, \tau\mathrm{-Risk}) - \kappa(\widehat{R\mathrm{-risk}}^*, \tau\mathrm{-Risk})$
    "kendalltau_stats__ref_r_risk": r"$\kappa (\ell, \tau\mathrm{{-Risk}}) - \kappa(\widehat{\mathrm{R-risk}}, \tau\mathrm{{-Risk}})$",
    "kendalltau_stats__ref_mean_risks": r"Relative $\kappa(\ell,\tau\mathrm{{-Risk}})$",
    "kendalltau_stats__ref_mean_risks_long": r"""Relative $\kappa(\ell,\tau\mathrm{{-Risk}})$ compared to mean over all metrics Kendall's""",
    "validation_ate": r"$\tau_{\mathcal{V}}$",
    "ate": r"$\tau_{\mathcal{S}}$",
}

METRIC_PALETTE = {
    CAUSAL_METRIC_LABELS["mu_risk"]: TAB_COLORS[10],
    CAUSAL_METRIC_LABELS["mu_iptw_risk"]: TAB_COLORS[0],
    CAUSAL_METRIC_LABELS["mu_ipw_n_risk"]: TAB_COLORS[18],
    CAUSAL_METRIC_LABELS["r_risk"]: TAB_COLORS[2],
    CAUSAL_METRIC_LABELS["u_risk"]: TAB_COLORS[4],
    CAUSAL_METRIC_LABELS["w_risk"]: TAB_COLORS[8],
    CAUSAL_METRIC_LABELS["r_risk_gold_e"]: TAB_COLORS[16],
    CAUSAL_METRIC_LABELS["r_risk_gold_m"]: TAB_COLORS[7],
    # CAUSAL_METRIC_LABELS["r_risk_ipw"]: TAB_COLORS[4],
    # CAUSAL_METRIC_LABELS["r_risk_IS2"]: TAB_COLORS[8],
    CAUSAL_METRIC_LABELS["oracle_mu_iptw_risk"]: TAB_COLORS[0],
    CAUSAL_METRIC_LABELS["oracle_r_risk"]: TAB_COLORS[2],
    CAUSAL_METRIC_LABELS["oracle_u_risk"]: TAB_COLORS[4],
    CAUSAL_METRIC_LABELS["oracle_w_risk"]: TAB_COLORS[8],
    # CAUSAL_METRIC_LABELS["oracle_r_risk_ipw"]: TAB_COLORS[5],
    # CAUSAL_METRIC_LABELS["oracle_r_risk_IS2"]: TAB_COLORS[9],
    CAUSAL_METRIC_LABELS["tau_risk"]: TAB_COLORS[6],
}

METRIC_PALETTE_BOX_PLOTS = {
    CAUSAL_METRIC_LABELS["mu_risk"]: TAB_COLORS[10],
    CAUSAL_METRIC_LABELS["mu_iptw_risk"]: TAB_COLORS[0],
    CAUSAL_METRIC_LABELS["mu_ipw_n_risk"]: TAB_COLORS[18],
    CAUSAL_METRIC_LABELS["r_risk"]: TAB_COLORS[2],
    CAUSAL_METRIC_LABELS["u_risk"]: TAB_COLORS[4],
    CAUSAL_METRIC_LABELS["w_risk"]: TAB_COLORS[8],
    CAUSAL_METRIC_LABELS["r_risk_gold_e"]: TAB_COLORS[16],
    CAUSAL_METRIC_LABELS["r_risk_gold_m"]: TAB_COLORS[7],
    # CAUSAL_METRIC_LABELS["r_risk_ipw"]: TAB_COLORS[4],
    # CAUSAL_METRIC_LABELS["r_risk_IS2"]: TAB_COLORS[8],
    CAUSAL_METRIC_LABELS["oracle_mu_iptw_risk"]: TAB_COLORS[1],
    CAUSAL_METRIC_LABELS["oracle_r_risk"]: TAB_COLORS[3],
    CAUSAL_METRIC_LABELS["oracle_u_risk"]: TAB_COLORS[5],
    CAUSAL_METRIC_LABELS["oracle_w_risk"]: TAB_COLORS[9],
    # CAUSAL_METRIC_LABELS["oracle_r_risk_ipw"]: TAB_COLORS[5],
    # CAUSAL_METRIC_LABELS["oracle_r_risk_IS2"]: TAB_COLORS[9],
    CAUSAL_METRIC_LABELS["tau_risk"]: TAB_COLORS[6],
}

METRIC_LS = {}
for metric in CAUSAL_METRICS:
    oracle_regex = metric.startswith("oracle") | (metric == "tau_risk")
    semi_oracle_regex = metric.find("gold") != -1
    metric_ls = "-"
    if oracle_regex:
        metric_ls = "--"
    elif semi_oracle_regex:
        metric_ls = "-."
    METRIC_LS[metric] = metric_ls

MMD_LABEL = "Control vs treated\noverlap violation in transformed basis (MMD)"
LEGEND_SELECTION_LABEL = r"""Causal metric $\ell$
for selection"""

METRIC_OF_INTEREST_LABELS = {
    "check_m_r2": r"$R2(\check{m})$",
    "check_e_bss": r"$BSS(\check{e})$",
    "check_e_eta": r"$min(\check{e}(x), 1 - \check{e}(x))$",
    "check_e_auroc": r"$AUROC(\check{e})$",
    "check_e_mse": r"$MSE(e*, \check{e})$",
    "check_e_IPW_mse": r"$MSE(ipw*, \check{ipw})$",
    "check_e_inv_mse": r"$MSE(\frac{1}{e*}, \frac{1}{\check{e}})$",
    "test_transformed_mmd": MMD_LABEL,
    "train_transformed_mmd": MMD_LABEL,
    "test_tv": "Overlap violation between control and treated (TV)",
    "d_tv": "Overlap violation between control and treated (TV)",
    "test_d_normalized_tv": "Control vs treated\noverlap violation (normalized TV)",
    "hat_d_normalized_tv": "Control vs treated\noverlap violation\n(estimated normalized TV)",
    "dgp_d_normalized_tv": "Overlap violation between control and treated \n(normalized TV)",
    "reweighted_surfaces_diff": r"$\frac{1}{n} \sum_{i=1}^{n} |\left(\mu_1(x_i) - \mu_0(x_i) \right) (2e(x_i) -1)|$",
    "train_tv": "Overlap violation between control and treated (TV)",
    "tau_risk_as_oracle": r"$\widehat{\tau Risk}(f^*_{\tau Risk})$",
    "chamfer_distance": r"Distance to true basis expansion D(\hat B, B^*)",
    "treatment_heterogeneity_norm": r"$std(\tau)/|ATE|$",
    "log_treatment_heterogeneity_norm": r"$log(std(\tau)/|ATE|)$",
    "treatment_heterogeneity": r"$std(\tau)$",
    "heterogeneity_score": r"$\mathcal{H}$",
    "heterogeneity_score_norm": r"$\mathcal{H}/|ATE|$",
    "effect_ratio": r"$\frac{1}{N} \sum_{i=1}^N | \frac{\mu_{1}(x_i) - \mu_{0}(x_i)}{\mu_{0}(x_i)}|$",
    "effect_ratio_sym": r"\frac{1}{N} \sum_{i=1}^N  \frac{|\mu_{1}(x_i) - \mu_{0}(x_i)|}{|\mu_{0}(x_i) + \mu_{1}(x_i) - \frac{1}{N} \sum_{i=j}^N\mu_{0}(x_j) + \mu_{1}(x_j)|}",
    "effect_ratio_sym2":r"\frac{1}{N} \sum_{i=1}^N\frac{\frac{1}{N} \sum_{i=1}^N |\mu_{1}(x_i) - \mu_{0}(x_i)|}{ |\mu_{0}(x_i) + \mu_{1}(x_i) - \frac{1}{N} \sum_{j=1}^N \mu_{0}(x_j) + \mu_{1}(x_j)|}", 
    "effect_variation": r"\hat_{var}(\mu_1 - \mu_0) / \hat_{var}(\mu_1+\mu_0)"
}

ORACLE_METRIC_NAMES = [
    metric
    for metric in CAUSAL_METRICS
    if (metric.lower().find("oracle") != -1)
    or (metric == "tau_risk")
    or (metric.lower().find("gold") != -1)
]
FEASIBLE_METRIC_NAMES = [
    metric for metric in CAUSAL_METRICS if metric not in ORACLE_METRIC_NAMES
]

METRIC_ORDER = [
    "mu_risk",
    "mu_iptw_risk",
    "oracle_mu_iptw_risk",
    "w_risk",
    "oracle_w_risk",
    "u_risk",
    "oracle_u_risk",
    "r_risk",
    "oracle_r_risk",
]

# Labels for the boxplots
OVERLAP_BIN_COL = "Overlap"
OVERLAP_BIN_LABELS = ["Strong Overlap", "Medium Overlap", "Weak Overlap"]
OVERLAP_BIN_PALETTE = dict(
    zip(
        OVERLAP_BIN_LABELS,
        sns.color_palette("YlOrRd", n_colors=len(OVERLAP_BIN_LABELS)),
    )
)
EFFECT_RATIO_BIN_COL = {
    "effect_ratio":"Causal effect ratio (not symmetric)",
    "effect_ratio_sym":"Causal effect ratio",
    "effect_ratio_sym2":"Causal effect ratio (variant)",
    "effect_variation":"Causal effect variation",
}
EFFECT_RATIO_BIN_LABELS = ["Low", "Medium", "High"]
EFFECT_RATIO_BIN_PALETTE = dict(
    zip(
        EFFECT_RATIO_BIN_LABELS,
        sns.color_palette("mako_r", n_colors=len(EFFECT_RATIO_BIN_LABELS)),
    )
)

METRIC_LABEL = "Metric"
NUISANCE_LABEL = "Nuisance models"
LINEAR_NUISANCES_LABEL = "Linear"
RFOREST_NUISANCES_LABEL = "Forests"
BOOSTING_NUISANCES_LABEL = "Boosting"
STACKED_NUISANCES_LABEL = "Stacked"
NUISANCE_PALETTE = {
    LINEAR_NUISANCES_LABEL: TAB_COLORS[4],
    STACKED_NUISANCES_LABEL: TAB_COLORS[8],
    RFOREST_NUISANCES_LABEL: TAB_COLORS[6],
    BOOSTING_NUISANCES_LABEL: TAB_COLORS[10],
}

SHARED_SET_LABEL = "Shared sets"
SEPARATED_SET_LABEL = "Separated sets"
PROCEDURE_LABEL = "Training procedure"
PROCEDURE_PALETTE = {
    SHARED_SET_LABEL: TAB_COLORS[0],
    SEPARATED_SET_LABEL: TAB_COLORS[2],
}

DATASET_LABEL = "Dataset"
CAUSSIM_LABEL = "Caussim\n (N=5 000)"
ACIC_16_LABEL = "ACIC 2016\n (N=4 802)"
ACIC_18_LABEL = "ACIC 2018 \n (N=5 000)"
TWINS_LABEL = "Twins \n (N= 11 984)"
TWINS_DS_LABEL = "Twins downsampled\n (N=4 794)"

MAP_DATASET_LABEL = {
    "caussim": CAUSSIM_LABEL,
    "acic_2016": ACIC_16_LABEL,
    "acic_2018": ACIC_18_LABEL,
    "twins": TWINS_LABEL,
}

DATASETS_PALETTE = {
    TWINS_LABEL: TAB_COLORS[18],
    TWINS_DS_LABEL: TAB_COLORS[19],
    ACIC_16_LABEL: TAB_COLORS[7],
    CAUSSIM_LABEL: TAB_COLORS[16],
    ACIC_18_LABEL: TAB_COLORS[9],
}


def plot_evaluation_metric(
    comparison_df_w_best_as_oracle: pd.DataFrame,
    nuisance_models_label: str = None,
    evaluation_metric: str = "normalized_bias_tau_risk_to_best_method",
    measure_of_interest_name: str = "d_normalized_tv",
    selection_metrics: List[str] = CAUSAL_METRICS,
    lowess_type: str = "seaborn",
    lowess_kwargs: Dict = None,
    ylog: bool = True,
    show_legend: bool = True,
    show_y_label: bool = True,
    plot_lowess_ci: bool = True,
    linewidth: float = 6,
) -> plt.Axes:
    available_lowess = [
        "seaborn",
        "lowess_quantile",
        "statsmodels",
        "scatter_only",
        "nonparametric_quantile",
    ]

    if evaluation_metric in [
        "normalized_bias_tau_risk_to_best_method",
        kendalltau_stats.__name__,
    ]:
        selection_metrics = [m for m in selection_metrics if m != "tau_risk"]
    # TODO: a np.percentile should be better
    log_clip = 1e-6
    q_label = None
    assert (
        lowess_type in available_lowess
    ), f"Choose a lowess type in {available_lowess}, got {lowess_type}"
    if lowess_kwargs is None:
        lowess_kwargs = {"frac": 0.66, "it": 10, "quantile": 0.5}
    # ### Evaluation metric of selection metrics against a chosen measure of interst ###
    legend_size = 20
    label_size = 40
    scatter_s = 20
    if show_legend == False:
        figsize = (14, 10)
    else:
        figsize = (10, 8)
    _, ax = plt.subplots(figsize=figsize)
    for metric in tqdm(selection_metrics):
        lowess_kwargs_ = lowess_kwargs.copy()
        label = CAUSAL_METRIC_LABELS[metric]
        color = METRIC_PALETTE[label]
        ls = METRIC_LS[metric]
        metric_data = comparison_df_w_best_as_oracle.loc[
            (comparison_df_w_best_as_oracle["causal_metric"] == metric)
        ]
        if ylog:
            metric_data.loc[:, evaluation_metric] = np.clip(
                metric_data[evaluation_metric], log_clip, np.infty
            )
        if lowess_type == "seaborn":
            sns.regplot(
                ax=ax,
                data=metric_data,
                x=measure_of_interest_name,
                y=evaluation_metric,
                color=color,
                ci=None,
                label=label,
                lowess=True,
                line_kws={"ls": ls, "lw": linewidth},
                scatter_kws={"s": scatter_s, "alpha": 0.5},
            )
        else:
            sns.scatterplot(
                ax=ax,
                data=metric_data,
                x=measure_of_interest_name,
                y=evaluation_metric,
                color=color,
                label=label,
                alpha=0.5,
                size=scatter_s,
                legend=False,
            )
            if lowess_type == "scatter_only":
                pass
            elif lowess_type == "statsmodels":
                # return its own grid
                xy_pred_lowess = lowess(
                    metric_data[evaluation_metric],
                    metric_data[measure_of_interest_name],
                    **lowess_kwargs_,
                ).T
                if ylog:
                    y_q = np.clip(xy_pred_lowess[1], log_clip, np.infty)
                x_q = xy_pred_lowess[0]
            elif lowess_type == "lowess_quantile":
                # Using [moepy implementation](https://ayrtonb.github.io/Merit-Order-Effect/ug-01-hydro-seasonality/)
                # Map to statsmodel api
                lowess_kwargs_["num_fits"] = lowess_kwargs_.pop("it", 10)
                quantiles2ix = {
                    np.round(q, 1): int(10 * q - 1) for q in np.arange(0.1, 1, 0.1)
                }
                target_quantile = lowess_kwargs_.pop("quantile", 0.5)
                if plot_lowess_ci:
                    qs = [0.05, target_quantile, 0.95]
                    q_index = 1
                else:
                    qs = [target_quantile]
                    q_index = 0
                logging.info(f"quantiles {qs}")
                quantile_df = moepy_lowess.quantile_model(
                    metric_data[measure_of_interest_name].values,
                    metric_data[evaluation_metric].values,
                    **lowess_kwargs_,
                    qs=qs,
                )
                if ylog:
                    y_q = np.clip(quantile_df.iloc[:, q_index], log_clip, np.infty)
                else:
                    y_q = quantile_df.iloc[:, q_index]
                x_q = quantile_df.index
                if plot_lowess_ci:
                    ax.fill_between(
                        quantile_df.index,
                        quantile_df.iloc[:, 0],
                        quantile_df.iloc[:, 2],
                        color=color,
                        alpha=0.1,
                    )
            elif lowess_type == "nonparametric_quantile":
                # From [sklearn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html?highlight=prediction%20intervals%20gradient%20boosting%20regression)
                q_ = lowess_kwargs_.pop("quantile", 0.5)
                q_label = q_
                all_models = {}
                common_params = dict(
                    learning_rate=0.05,
                    n_estimators=200,
                    max_depth=2,
                    min_samples_leaf=9,
                    min_samples_split=9,
                )
                x_q = np.atleast_2d(
                    np.linspace(
                        metric_data[measure_of_interest_name].min(),
                        metric_data[measure_of_interest_name].max(),
                        1000,
                    )
                ).T

                gbr = GradientBoostingRegressor(
                    loss="quantile", alpha=q_, **common_params
                )
                all_models[f"q {q_:1.2f}"] = gbr.fit(
                    np.atleast_2d(metric_data[measure_of_interest_name]).T,
                    metric_data[evaluation_metric],
                )
                if ylog:
                    y_q = np.clip(
                        all_models[f"q {q_:1.2f}"].predict(x_q), log_clip, np.infty
                    )
            ax.plot(
                x_q,
                y_q,
                color=color,
                linewidth=linewidth,
                linestyle=ls,
            )
    if show_legend:
        leg_handles, leg_labels = get_selection_legend(selection_metrics)
        legend_title = "Selection metric"
        if q_label is not None:
            legend_title += f"\n {q_label} quantile"
        ax.add_artist(
            plt.legend(
                title=legend_title,
                handles=leg_handles,
                labels=leg_labels,
                bbox_to_anchor=(1.01, 1),
                borderaxespad=0,
                prop={
                    "size": legend_size,
                },
                ncol=2,
            )
        )
    ax.set_xlabel(
        xlabel=METRIC_OF_INTEREST_LABELS[measure_of_interest_name], fontsize=label_size
    )
    if show_y_label:
        ax.set_ylabel(
            ylabel=EVALUATION_METRIC_LABELS[evaluation_metric], fontsize=label_size
        )
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=20)
    if nuisance_models_label is not None:
        ax.text(
            0.01,
            1.1,
            f"{nuisance_models_label.capitalize()} nuisance models",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            # transform=plt.gcf().transFigure,
        )
    if ylog:
        ax.set(yscale="log")
    return ax


def plot_ranking_aggregation(
    rankings_aggregation: pd.DataFrame,
    expe_indices: List[str],
    x_metric_name: str = "test_d_normalized_tv",
    nuisance_models_label: str = None,
    lowess_type="lowess_quantile",
    lowess_kwargs: Dict = None,
    y_lim: Tuple[float, float] = (0, 1),
    show_legend: bool = True,
    show_y_label: bool = True,
    reference_metric: str = None,
) -> plt.Axes:
    """For a given dataset family ie. a combination of different triplets (dataset, random seed, overlap setup), plot the kendall's tau score vs the measure of overlap, for each causal metric registered in the rankings_aggregation dataframe.
    Make a lowess approximation of the curves.

    Parameters
    ----------
    rankings_aggregation : pd.DataFrame
        _description_

    reference_metric : Metric of reference to which compare all other metrics. This is thought as the best ranking that we can have from all studied causal metrics.

    """
    aggregation_f_name = kendalltau_stats.__name__

    rankings_matches = [
        re.search(f"{aggregation_f_name}__tau_risk_(.*)", col)
        for col in rankings_aggregation.columns
    ]
    selection_metrics = [
        reg_match.group(1) for reg_match in rankings_matches if reg_match is not None
    ]
    rankings_name = [
        reg_match.group(0) for reg_match in rankings_matches if reg_match is not None
    ]
    if reference_metric is not None:
        reference_ranking_name = f"{aggregation_f_name}__tau_risk_{reference_metric}"
        if reference_metric in selection_metrics:
            selection_metrics.remove(reference_metric)
            rankings_name.remove(reference_ranking_name)
        elif reference_metric == "mean_risks":
            rankings_aggregation[reference_ranking_name] = rankings_aggregation[
                rankings_name
            ].mean(axis=1)
        else:
            raise ValueError(
                f"reference_metric should be in {selection_metrics} or 'mean_risks', got {reference_metric}"
            )

        for ranking_ in rankings_name:
            rankings_aggregation[ranking_] = (
                rankings_aggregation[ranking_]
                - rankings_aggregation[reference_ranking_name]
            )
        evaluation_metric = aggregation_f_name + "__ref_" + reference_metric
    else:
        evaluation_metric = aggregation_f_name

    var_name = "causal_metric"

    if x_metric_name not in expe_indices:
        raise ValueError("Metric of interest (x variable) should be in expe_indices.")
    rankings_aggregation_melted = rankings_aggregation.melt(
        id_vars=expe_indices,
        value_vars=rankings_name,
        var_name=var_name,
        value_name=evaluation_metric,
    )
    rankings_aggregation_melted = rankings_aggregation_melted.assign(
        **{
            var_name: lambda df: df[var_name].str.replace(
                f"{aggregation_f_name}__tau_risk_", ""
            )
        }
    )

    ax = plot_evaluation_metric(
        comparison_df_w_best_as_oracle=rankings_aggregation_melted,
        evaluation_metric=evaluation_metric,
        measure_of_interest_name=x_metric_name,
        nuisance_models_label=nuisance_models_label,
        selection_metrics=selection_metrics,
        lowess_type=lowess_type,
        lowess_kwargs=lowess_kwargs,
        ylog=False,
        show_legend=show_legend,
        show_y_label=show_y_label,
    )
    ax.set(ylim=y_lim)
    return ax


def plot_metric_rankings_by_overlap_bin(
    expe_results: Dict[str, pd.DataFrame],
    reference_metric: str,
    expe_causal_metrics: List[str],
    comparison_label: str,
    plot_middle_overlap_bin: bool = False,
):
    """Box plot of the ranking of the causal metrics for each overlap bin.
    - the hue is defined by the user in the keys of the expe_results,
    - the x axis is the evaluation metric (kendall's tau)
    - the y axis is either "metric" or "dataset_name" if there is only one metric
      selected.

    Parameters
    ----------
    expe_results : Dict[str, pd.DataFrame]
        _description_
    reference_metric : str
        _description_
    expe_causal_metrics : List[str]
        _description_
    comparison_label : str
        _description_
    plot_middle_overlap_bin : bool, optional
        _description_, by default False
    y_axis : str, optional
        _description_, by default "metric"

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    aggregation_f_name = kendalltau_stats.__name__
    binned_results = []

    for xp_name, xp_res in expe_results.items():
        kendall_by_overlap_bin, evaluation_metric = get_kendall_by_overlap_bin(
            xp_res=xp_res,
            expe_causal_metrics=expe_causal_metrics,
            reference_metric=reference_metric,
            plot_middle_overlap_bin=plot_middle_overlap_bin,
        )
        kendall_by_overlap_bin[comparison_label] = xp_name
        binned_results.append(kendall_by_overlap_bin)
    binned_results_df = pd.concat(binned_results, axis=0)
    if plot_middle_overlap_bin:
        col_order = [
            OVERLAP_BIN_LABELS[0],
            OVERLAP_BIN_LABELS[1],
            OVERLAP_BIN_LABELS[2],
        ]
    else:
        col_order = [OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[2]]
    if comparison_label == NUISANCE_LABEL:
        palette = NUISANCE_PALETTE
    elif comparison_label == PROCEDURE_LABEL:
        palette = PROCEDURE_PALETTE
    metric_order = [
        CAUSAL_METRIC_LABELS[metric_label_]
        for metric_label_ in METRIC_ORDER
        if CAUSAL_METRIC_LABELS[metric_label_]
        in binned_results_df[METRIC_LABEL].unique()
    ]
    g = sns.catplot(
        data=binned_results_df,
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=metric_order,
        hue=comparison_label,
        col=OVERLAP_BIN_COL,
        aspect=1.2,
        height=10,
        kind="box",
        palette=palette,
        col_order=col_order,
    )
    # Aesthetics
    g.set_titles(col_template="{col_name}")
    g.set(xlabel="", ylabel="")
    # g.fig.suptitle(EVALUATION_METRIC_LABELS[evaluation_metric], y=0.04)
    g.fig.suptitle(EVALUATION_METRIC_LABELS[evaluation_metric + "_long"], y=0.04)
    return g


def get_kendall_by_overlap_bin(
    xp_res: pd.DataFrame,
    expe_causal_metrics: List[str],
    reference_metric: str = None,
    plot_middle_overlap_bin: bool = True,
) -> Tuple[pd.DataFrame, str]:
    """Group by overlap bin a kendall with reference to the kendall of a given metric

    Parameters
    ----------
    xp_res : pd.DataFrame
        _description_
    reference_metric : str
        _description_
    expe_causal_metrics : List[str]
        _description_
    plot_middle_overlap_bin : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    aggregation_f_name = kendalltau_stats.__name__
    candidate_params = get_candidate_params(xp_res)
    overlap_measure, expe_indices = get_expe_indices(xp_res)
    expe_rankings = get_metric_rankings_by_dataset(
        expe_results=xp_res,
        expe_indices=expe_indices,
        causal_metrics=expe_causal_metrics,
        candidate_params=candidate_params,
    )

    rankings_agg = get_rankings_aggregate(
        expe_rankings=expe_rankings,
        expe_indices=expe_indices,
        causal_metrics=expe_causal_metrics,
    )

    rankings_matches = [
        re.search(f"{aggregation_f_name}__tau_risk_(.*)", col)
        for col in rankings_agg.columns
    ]
    selection_metrics = [
        reg_match.group(1) for reg_match in rankings_matches if reg_match is not None
    ]
    rankings_name = [
        reg_match.group(0) for reg_match in rankings_matches if reg_match is not None
    ]

    if reference_metric is not None:
        reference_ranking_name = f"{aggregation_f_name}__tau_risk_{reference_metric}"
        if reference_metric in selection_metrics:
            selection_metrics.remove(reference_metric)
            rankings_name.remove(reference_ranking_name)
        elif reference_metric == "mean_risks":
            rankings_agg[reference_ranking_name] = rankings_agg[rankings_name].mean(
                axis=1
            )
        else:
            raise ValueError(
                f"reference_metric should be in {selection_metrics} or 'mean_risks', got {reference_metric}"
            )

        for ranking_ in rankings_name:
            rankings_agg[ranking_] = (
                rankings_agg[ranking_] - rankings_agg[reference_ranking_name]
            )
        evaluation_metric = aggregation_f_name + "__ref_" + reference_metric
    else:
        evaluation_metric = aggregation_f_name
    rankings_aggregation_melted = rankings_agg.melt(
        id_vars=expe_indices,
        value_vars=rankings_name,
        var_name=METRIC_LABEL,
        value_name=evaluation_metric,
    )
    # shape = n_experiences x n_causal_metrics
    bins_quantiles = [0, 0.33, 0.66, 1]
    bins_values = (
        rankings_aggregation_melted[overlap_measure].quantile(bins_quantiles).values
    )
    # bins_labels = [f"{b_low:.2f}-{b_sup:.2f}" for b_low, b_sup in
    # zip(bins_values[:-1], bins_values[1:])]
    rankings_aggregation_melted[OVERLAP_BIN_COL] = pd.cut(
        rankings_aggregation_melted[overlap_measure],
        bins=bins_values,
        labels=OVERLAP_BIN_LABELS,
    ).astype(str)
    # keep only extrem tertiles
    if plot_middle_overlap_bin:
        kept_bins = [
            OVERLAP_BIN_LABELS[0],
            OVERLAP_BIN_LABELS[1],
            OVERLAP_BIN_LABELS[2],
        ]
    else:
        kept_bins = [OVERLAP_BIN_LABELS[0], OVERLAP_BIN_LABELS[2]]
    rankings_aggregation_melted = rankings_aggregation_melted.loc[
        rankings_aggregation_melted[OVERLAP_BIN_COL].isin(kept_bins)
    ]
    # adding type of metrics: feasible vs. semi-oracle
    rankings_aggregation_melted[METRIC_TYPE] = rankings_aggregation_melted[
        METRIC_LABEL
    ].apply(
        lambda x: SEMI_ORACLE_LABEL
        if (re.search("oracle", x) is not None)
        else FEASIBLE_LABEL
    )
    rankings_aggregation_melted[METRIC_LABEL] = rankings_aggregation_melted[
        METRIC_LABEL
    ].apply(
        lambda x: CAUSAL_METRIC_LABELS[
            re.sub(f"{aggregation_f_name}__tau_risk_", "", x)
        ]
    )
    return rankings_aggregation_melted, evaluation_metric


def get_selection_legend(metrics, subtitles=True):
    oracle_metrics_subset = [
        metric for metric in metrics if metric in ORACLE_METRIC_NAMES
    ]
    selection_oracle_legend_handles, selection_oracle_legend_labels = (
        [
            Line2D(
                [0],
                [0],
                color=METRIC_PALETTE[CAUSAL_METRIC_LABELS[metric]],
                lw=4,
                ls=METRIC_LS[metric],
            )
            for metric in oracle_metrics_subset
        ],
        [CAUSAL_METRIC_LABELS[metric] for metric in oracle_metrics_subset],
    )

    feasible_metrics_subset = [
        metric for metric in metrics if metric in FEASIBLE_METRIC_NAMES
    ]
    selection_feasible_legend_handles, selection_feasible_legend_labels = (
        [
            Line2D(
                [0],
                [0],
                color=METRIC_PALETTE[CAUSAL_METRIC_LABELS[metric]],
                lw=4,
                ls=METRIC_LS[metric],
            )
            for metric in feasible_metrics_subset
        ],
        [CAUSAL_METRIC_LABELS[metric] for metric in feasible_metrics_subset],
    )
    if subtitles:
        selection_legend_handles = [
            Line2D([0], [0], color="white"),
            *selection_oracle_legend_handles,
            Line2D([0], [0], color="white"),
            Line2D([0], [0], color="white"),
            *selection_feasible_legend_handles,
        ]
        selection_legend_labels = [
            "Oracle",
            *selection_oracle_legend_labels,
            "",
            "Feasible",
            *selection_feasible_legend_labels,
        ]
    else:
        selection_legend_handles = [
            *selection_oracle_legend_handles,
            *selection_feasible_legend_handles,
        ]
        selection_legend_labels = [
            *selection_oracle_legend_labels,
            *selection_feasible_legend_labels,
        ]
    return selection_legend_handles, selection_legend_labels


def plot_metric_legend(
    metrics: List[str], include_tau_risk: bool = False, ncol: int = 2
):
    oracle_metrics = [
        m for m in metrics if re.search("oracle|gold|tau_risk", m) is not None
    ]
    if not include_tau_risk:
        oracle_metrics = [m for m in oracle_metrics if m != "tau_risk"]
    feasible_metrics = [
        m for m in metrics if re.search("oracle|gold|tau_risk", m) is None
    ]

    o_handles, o_labels = get_selection_legend(oracle_metrics, subtitles=False)
    f_handles, f_labels = get_selection_legend(feasible_metrics, subtitles=False)

    if ncol == 1:
        figsize = (6, 14)
    else:
        figsize = (11, 9)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # plt.title('Selection metrics', loc="center", ha='center', va='center')
    oracle_title = "Semi-oracle"
    if "tau_risk" in oracle_metrics:
        if ncol == 1:
            oracle_title += "\n and oracle"
        else:
            oracle_title = "oracle"
    ax.add_artist(
        plt.legend(
            o_handles,
            o_labels,
            title=oracle_title,
            # bbox_to_anchor=(0, -0.5),
            loc="upper left",
            ncol=ncol,
        )
    )
    ax.add_artist(
        plt.legend(
            f_handles,
            f_labels,
            title="Feasible",
            # bbox_to_anchor=(0, 2),
            loc="lower left",
            ncol=ncol,
        )
    )
    ax.grid(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    # ax.xaxis.set_major_locator(ticker.NullLocator())
    # ax.yaxis.set_major_locator(ticker.NullLocator())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    return ax


def create_legend_for_candidates(candidates_family, colormap=None):
    """Create legend for candidates family, return also the candidates family colormap"""
    # get an approriate colormap
    if colormap is None:
        if len(candidates_family) <= 20:
            colormap = [cm.tab20(i) for i in range(len(candidates_family))]  # type:ignore
        else:
            colormap = plt.cm.get_cmap("viridis", len(candidates_family))  #  type:ignore

    candidates_colormap = {
        candidate_id: c for (candidate_id, c) in zip(candidates_family, colormap)
    }

    handles = []
    labels = []
    for candidate_id in candidates_family:
        handles.append(
            Line2D(
                [0],
                [0],
                color=candidates_colormap[candidate_id],
                marker="o",
                markersize=15,
                linestyle="None",
            )
        )
        model = "".join(candidate_id.split("__")[0])
        model = (
            model.replace("hist_gradient_boosting", "Boosted trees")
            .replace("random_forest", "Forests")
            .capitalize()
        )

        metalearner = "".join(candidate_id.split("__")[1]).replace(
            "meta_learner_name_", ""
        )
        params = candidate_id.split("__")[2:]
        params = [
            x for x in params if (x.find("nan") == -1 & x.find("final_estimator") == -1)
        ]
        # regex for smaller params names
        float_regex = "(max_depth_[0-9]+)[.][0-9]"
        params = [re.sub(float_regex, r"\1", x) for x in params]
        params = [x.replace("max_depth", "max depth") for x in params]
        params = [x.replace("learning_rate", "learning rate") for x in params]
        params = [x.replace("alpha", "penalty") for x in params]

        params = [x.replace("_", "=") for x in params]
        if model == "Boosted trees":
            labels.append(f"{model}, {' '.join(params)}")
        elif model == "Ridge":
            params = [""]
            if metalearner == "SLearner":
                labels.append(f"{model}")  # wo. interaction{' '.join(params)}")
            else:
                labels.append(f"{model} w. interaction{' '.join(params)}")
        else:
            labels.append(f"{model}, {' '.join(params)}")
    return (handles, labels), candidates_colormap


def plot_agreement_w_tau_risk(
    comparison_df_w_best_as_oracle: pd.DataFrame,
    overlap_measure: str = "test_d_normalized_tv",
    nuisance_models_label: str = None,
    n_bins: int = 10,
    show_overlap_distribution: bool = False,
    show_legend: bool = True,
    linewidth=6,
):
    """From a comparison dataframe, plot the agreement with tau risk
    Comparison dataframe contains top estimator for each pair (dataset, causal_metric)

    Parameters
    ----------
    comparison_df_w_best_as_oracle : pd.DataFrame
        _description_
    overlap_measure : str, optional
        _description_, by default "test_d_normalized_tv"
    nuisance_models_label : str, optional
        _description_, by default None
    ax : _type_, optional
        _description_, by default None
    n_bins : int, optional
        _description_, by default 10

    Returns
    -------
    _type_
        _description_
    """
    fig = plt.figure(figsize=(14, 7))
    if show_overlap_distribution:
        gs = fig.add_gridspec(
            2, 1, height_ratios=(2, 7), bottom=0.1, top=0.9, hspace=0.05
        )
        ax = fig.add_subplot(gs[1, 0])
        ax_histx = fig.add_subplot(gs[0, 0])
    else:
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax_histx = None
    aggreement_label = "Agree w tau_risk (%)"
    comparison_df_w_best_as_oracle_ = comparison_df_w_best_as_oracle.copy()
    # TODO: we could better than exact match (eg. kendall's tau) but need to have the full ordering of the methods and not only the best method.
    comparison_df_w_best_as_oracle_[aggreement_label] = (
        comparison_df_w_best_as_oracle_["tau_risk"]
        == comparison_df_w_best_as_oracle_["tau_risk_as_oracle"]
    )
    overlap_splitter = list(
        np.percentile(
            comparison_df_w_best_as_oracle_[overlap_measure], np.arange(0, 100, n_bins)
        )
    ) + [np.max(comparison_df_w_best_as_oracle_[overlap_measure])]
    p_dist_medians = [
        (overlap_splitter[i + 1] - overlap_splitter[i]) / 2 + overlap_splitter[i]
        for i in range(len(overlap_splitter) - 1)
    ]
    comparison_df_w_best_as_oracle_["bin_overlap"] = pd.cut(
        comparison_df_w_best_as_oracle_[overlap_measure],
        bins=overlap_splitter,
        labels=p_dist_medians,
    ).reset_index(drop=True)

    n_found_oracles_grouped_overlap = (
        comparison_df_w_best_as_oracle_.groupby(["bin_overlap", "causal_metric"])[
            aggreement_label
        ]
        .agg(["mean", "count"])
        .reset_index()
    )
    n_found_oracles_grouped_overlap["mean"] = (
        n_found_oracles_grouped_overlap["mean"] * 100
    )
    n_found_oracles_grouped_overlap["bin_overlap_numeric"] = np.round(
        n_found_oracles_grouped_overlap["bin_overlap"].astype(float), 5
    )
    n_found_oracles_grouped_overlap = n_found_oracles_grouped_overlap.loc[
        ~n_found_oracles_grouped_overlap["causal_metric"].isin(
            ["tau_risk", "mse_ate", "random"]
        )
    ]
    causal_metrics_measured = n_found_oracles_grouped_overlap["causal_metric"].unique()
    for causal_metric in causal_metrics_measured:
        label = CAUSAL_METRIC_LABELS[causal_metric]
        g = sns.lineplot(
            ax=ax,
            x="bin_overlap_numeric",
            y="mean",
            data=n_found_oracles_grouped_overlap.loc[
                n_found_oracles_grouped_overlap["causal_metric"] == causal_metric
            ],
            color=METRIC_PALETTE[label],
            label=label,
            linestyle=METRIC_LS[causal_metric],
            legend=False,
            linewidth=linewidth,
            markers=True,
        )
    if show_overlap_distribution:
        # TODO: not working : does not plot aligned with main plot
        sns.barplot(
            ax=ax_histx,
            y=n_found_oracles_grouped_overlap.loc[
                n_found_oracles_grouped_overlap["causal_metric"] == causal_metric
            ]["count"],
            x=n_found_oracles_grouped_overlap.loc[
                n_found_oracles_grouped_overlap["causal_metric"] == causal_metric
            ]["bin_overlap_numeric"],
            color="grey",
        )
        # Hide axes labels
        plt.setp(ax_histx.get_xticklabels(), visible=False)  # type:ignore
    max_agreement = n_found_oracles_grouped_overlap["mean"].max()
    min_agreement = n_found_oracles_grouped_overlap["mean"].min()

    ax.set(ylim=(max_agreement, min_agreement))
    xlabel = METRIC_OF_INTEREST_LABELS[overlap_measure]
    if show_legend == False:
        xlabel = re.sub("\\n", " ", xlabel)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Agreement with $\widehat{\tau\mathrm{-risk}}$ (%)")
    leg_handles, leg_labels = get_selection_legend(causal_metrics_measured)
    if show_legend:
        plt.legend(
            leg_handles, leg_labels, bbox_to_anchor=(1.01, 1), prop={"size": 20}, ncol=2
        )
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    if nuisance_models_label is not None:
        ax.text(
            0,
            1.1,
            f"{nuisance_models_label.capitalize()} nuisance models",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
        )
    return ax, n_found_oracles_grouped_overlap


# ### Normalized Total Variation reconstruction ### #
def plot_diff_between_n_tv_and_approximation(
    n_tv_results,
    calibration: bool = False,
    x_colname: str = "oracle_n_tv",
    save: bool = True,
):
    """
    From experimental results from `caussim/experiences/normalized_total_variation_approximation.py` plot the difference between the oracle and the approximation of normalized Total Variation versus some metric of interest among those registered : bss, bs, roc_auc, n_tv.
    Args:
        n_tv_results (_type_): _description_
        calibration (bool, optional): _description_. Defaults to False.
        x_colname (str, optional): _description_. Defaults to "oracle_n_tv".
        save (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    color_models = {
        "histgradientboostingclassifier": TAB_COLORS[0],
        "logisticregression": TAB_COLORS[2],
    }
    label_models = {
        "histgradientboostingclassifier": "HGB",
        "logisticregression": "LR",
    }
    datasets = n_tv_results["dataset_name"].unique()
    n_tv_results[f"n_tv_diff"] = n_tv_results["oracle_n_tv"] - n_tv_results["n_tv"]
    label_xaxis = {
        "oracle_n_tv": "Oracle normalized \n Total Variation",
        "bss": "Classifier \n Brier Skill Score",
        "bs": "Classifier \n Brier Score",
        "roc_auc": "ROC AUC",
    }

    fig, axs = plt.subplots(1, len(datasets), figsize=(12, 4), sharey=True)
    if len(datasets) == 1:
        axs = [axs]
    for i, dataset_name in enumerate(datasets):
        for model in color_models.keys():
            n_tv_model = n_tv_results.loc[
                (n_tv_results["model_name"] == model)
                & (n_tv_results["dataset_name"] == dataset_name)
            ]
            X_unsorted = n_tv_model[x_colname].values
            sorted_ix = np.argsort(X_unsorted)
            X = X_unsorted[sorted_ix]
            y = n_tv_model[f"n_tv_diff"].values[sorted_ix]
            quantiles = [0.95, 0.5, 0.05]
            hist_quantiles = {
                f"quantile={quantile:.2f}": GradientBoostingRegressor(
                    loss="quantile", alpha=quantile
                ).fit(np.atleast_2d(X).transpose(), y)
                for quantile in quantiles
            }

            axs[i].plot(
                X, y, "o", alpha=0.3, c=color_models[model], label=color_models[model]
            )

            axs[i].plot(
                X,
                hist_quantiles["quantile=0.50"].predict(np.atleast_2d(X).transpose()),
                label=model,
                c=color_models[model],
            )

            axs[i].fill_between(
                X,
                hist_quantiles["quantile=0.05"].predict(np.atleast_2d(X).transpose()),
                hist_quantiles["quantile=0.95"].predict(np.atleast_2d(X).transpose()),
                color=color_models[model],
                alpha=0.2,
            )
            if i == 0:
                axs[i].set(
                    ylabel="Difference between \n oracle and approximation of nTV"
                )
            axs[i].set(xlabel=label_xaxis[x_colname], title=dataset_name)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_models[model],
            label=label_models[model],
            markersize=15,
            alpha=0.5,
        )
        for model in color_models.keys()
    ]
    if calibration:
        title = "Calibrated \n Classifiers"
    else:
        title = "Classifiers"
    plt.legend(title=title, handles=legend_elements, bbox_to_anchor=(1, 1))
    if save:
        plt.savefig(
            DIR2NOTES
            / "n_tv_approximation"
            / f"diff_oracle_ntv_to_n_tv_calibration={calibration}_vs_{x_colname}.pdf",
            bbox_inches="tight",
        )
    plt.show()
    return fig


# ### Functions focused on propensity scores ### #


def plot_logit(sample: pd.DataFrame, fig, ax):
    df = sample.copy()
    df["logit(e)"] = np.log(1 - df["e"]) - np.log(df["e"])
    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # sns.kdeplot(ax=axes[0], data=df, x="logit(e)", color="grey", linestyle="--")
    sns.histplot(
        ax=ax,
        data=df,
        x="logit(e)",
        hue="a",
        kde=True,
        legend=False,
        palette=COLOR_MAPPING,
        stat="probability",
        common_norm=False,
        bins=100,
    )
    ax.set_title(
        "Distribution of the logit of treatment propensity \n by treatment status"
    )
    return fig, ax


def get_data_from_facetgrid_boxplot(
    data, x, y, hue, col, col_order, hue_order=None, order=None
) -> pd.DataFrame:
    # if I want to bold extrem values:
    # https://flopska.com/highlighting-pandas-to_latex-output-in-bold-face-for-extreme-values.html
    box_plot_df = (
        data.groupby([col, y, hue])
        .agg(
            **{
                "Median": pd.NamedAgg(column=x, aggfunc=lambda x: np.median(x)),
                "q25": pd.NamedAgg(column=x, aggfunc=lambda x: np.quantile(x, 0.25)),
                "q75": pd.NamedAgg(column=x, aggfunc=lambda x: np.quantile(x, 0.75)),
            }
        )
        .reset_index()
    )
    for colname, order_ in zip([hue, y, col], [hue_order, order, col_order]):
        if order_ is not None:
            box_plot_df[colname] = pd.Categorical(
                box_plot_df[colname], categories=order_, ordered=True
            )
    sort_combined = [col, y, hue]
    box_plot_df.sort_values(by=sort_combined, inplace=True)
    box_plot_df["IQR"] = box_plot_df["q75"] - box_plot_df["q25"]
    box_plot_df.drop(["q75", "q25"], axis=1, inplace=True)
    full_table = []
    for col_value in col_order:
        col_table = (
            box_plot_df.loc[box_plot_df[col] == col_value]
            .drop(col, axis=1)
            .set_index([y, hue])
        )
        full_table.append(col_table)

    box_plot_df_two_column = pd.concat(full_table, keys=col_order, axis=1)
    return box_plot_df_two_column


def plot_overlap(
    sample: pd.DataFrame,
    overlap: str = "",
    treatment_assignment: str = "",
    random_seed: str = "",
    fig=None,
    ax=None,
):
    """Plot distributions of propensity logit for each treatment status (ie. each population)

    Args:
        sample (pd.DataFrame): _description_
        overlap (str, optional): _description_. Defaults to "".
        treatment_assignment (str, optional): _description_. Defaults to "".
        random_seed (str, optional): _description_. Defaults to "".
        fig (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if fig is None:
        save = True
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    else:
        save = False
    df = sample.copy()
    df["logit(e)"] = np.log(1 - df["e"]) - np.log(df["e"])

    # plot distributions of logit depending on treatment:
    mask_treated = df["a"] == 1
    mask_control = df["a"] == 0
    # plot distributions of e(x) depending on treatment:
    sns.histplot(
        ax=ax,
        data=df,
        x="e",
        hue="a",
        kde=True,
        palette=COLOR_MAPPING,
        stat="probability",
        bins=100,
        common_norm=False,
    )
    legend_handles = ax.legend_.legendHandles  # type:ignore
    legend_labels = [LABEL_MAPPING[float(txt.get_text())] for txt in ax.legend_.texts]  # type:ignore
    plt.legend(
        legend_handles,
        legend_labels,
        title=r"Treatment Status",
        borderaxespad=0,
        ncol=1,
        loc="upper right",  # bbox_to_anchor=(0.9, 1.01),
    )
    m_stats_propensity, p_value_propensity = mannwhitneyu(
        df.loc[mask_treated, "e"],
        df.loc[mask_control, "e"],
    )
    ps_title = "Distribution of the propensity scores by treatment status \n Man-Whitney p-value={:.2E}".format(
        p_value_propensity
    )
    sup_title = (
        f"Overlap parameter={overlap}, treatment assignment={treatment_assignment}"
    )

    ps_title = ps_title + "\n" + sup_title
    ax.set_title(ps_title)
    plt.tight_layout()
    if save:
        fig.savefig(
            DIR2FIGURES
            / f"overlap_measure_overlap={overlap}_treatment_assigment={treatment_assignment}_seed={random_seed}.png",
            bbox_inches="tight",
        )
    return fig, ax


def get_kendall_by_effect_ratio_bin(
    xp_res: pd.DataFrame,
    expe_causal_metrics: List[str],
    reference_metric: str = None,
    plot_middle_overlap_bin: bool = True,
    measure_of_interest: str = "overlap",
) -> Tuple[pd.DataFrame, str]:
    """Group by effect ratio bin a kendall with reference to the kendall of a given metric

    Parameters
    ----------
    xp_res : pd.DataFrame
        _description_
    reference_metric : str
        _description_
    expe_causal_metrics : List[str]
        _description_
    plot_middle_overlap_bin : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    aggregation_f_name = kendalltau_stats.__name__
    candidate_params = get_candidate_params(xp_res)
    effect_ratio_measure, expe_indices = get_expe_indices(
        xp_res, measure_of_interest=measure_of_interest
    )

    expe_rankings = get_metric_rankings_by_dataset(
        expe_results=xp_res,
        expe_indices=expe_indices,
        causal_metrics=expe_causal_metrics,
        candidate_params=candidate_params,
    )

    rankings_agg = get_rankings_aggregate(
        expe_rankings=expe_rankings,
        expe_indices=expe_indices,
        causal_metrics=expe_causal_metrics,
    )

    rankings_matches = [
        re.search(f"{aggregation_f_name}__tau_risk_(.*)", col)
        for col in rankings_agg.columns
    ]
    selection_metrics = [
        reg_match.group(1) for reg_match in rankings_matches if reg_match is not None
    ]
    rankings_name = [
        reg_match.group(0) for reg_match in rankings_matches if reg_match is not None
    ]

    if reference_metric is not None:
        reference_ranking_name = f"{aggregation_f_name}__tau_risk_{reference_metric}"
        if reference_metric in selection_metrics:
            selection_metrics.remove(reference_metric)
            rankings_name.remove(reference_ranking_name)
        elif reference_metric == "mean_risks":
            rankings_agg[reference_ranking_name] = rankings_agg[rankings_name].mean(
                axis=1
            )
        else:
            raise ValueError(
                f"reference_metric should be in {selection_metrics} or 'mean_risks', got {reference_metric}"
            )

        for ranking_ in rankings_name:
            rankings_agg[ranking_] = (
                rankings_agg[ranking_] - rankings_agg[reference_ranking_name]
            )
        evaluation_metric = aggregation_f_name + "__ref_" + reference_metric
    else:
        evaluation_metric = aggregation_f_name
    rankings_aggregation_melted = rankings_agg.melt(
        id_vars=expe_indices,
        value_vars=rankings_name,
        var_name=METRIC_LABEL,
        value_name=evaluation_metric,
    )
    # shape = n_experiences x n_causal_metrics
    bins_quantiles = [0, 0.33, 0.66, 1]
    bins_values = (
        rankings_aggregation_melted[effect_ratio_measure]
        .quantile(bins_quantiles)
        .values
    )
    bins_labels = [
        f"{b_low:.2f}-{b_sup:.2f}"
        for b_low, b_sup in zip(bins_values[:-1], bins_values[1:])
    ]
    print("Bins labels", bins_labels)
    rankings_aggregation_melted[EFFECT_RATIO_BIN_COL[measure_of_interest]] = pd.cut(
        rankings_aggregation_melted[effect_ratio_measure],
        bins=bins_values,
        labels=EFFECT_RATIO_BIN_LABELS,
    ).astype(str)
    # keep only extrem tertiles
    if plot_middle_overlap_bin:
        kept_bins = [
            EFFECT_RATIO_BIN_LABELS[0],
            EFFECT_RATIO_BIN_LABELS[1],
            EFFECT_RATIO_BIN_LABELS[2],
        ]
    else:
        kept_bins = [EFFECT_RATIO_BIN_LABELS[0], EFFECT_RATIO_BIN_LABELS[2]]
    rankings_aggregation_melted = rankings_aggregation_melted.loc[
        rankings_aggregation_melted[EFFECT_RATIO_BIN_COL[measure_of_interest]].isin(kept_bins)
    ]
    # adding type of metrics: feasible vs. semi-oracle
    rankings_aggregation_melted[METRIC_TYPE] = rankings_aggregation_melted[
        METRIC_LABEL
    ].apply(
        lambda x: SEMI_ORACLE_LABEL
        if (re.search("oracle", x) is not None)
        else FEASIBLE_LABEL
    )
    rankings_aggregation_melted[METRIC_LABEL] = rankings_aggregation_melted[
        METRIC_LABEL
    ].apply(
        lambda x: CAUSAL_METRIC_LABELS[
            re.sub(f"{aggregation_f_name}__tau_risk_", "", x)
        ]
    )
    return rankings_aggregation_melted, evaluation_metric


def plot_kendall_compare_vs_measure(
    expe_results: pd.DataFrame,
    reference_metric: str,
    expe_causal_metrics: List[str],
    measure_of_interest: str = "overlap",
    quantile: float = 0.5,
    ylim_ranking: Tuple[float, float] = (-1.0, 1.0),
):
    """Plot the kendall tau between the reference metric and the other metrics
    vs the overlap.

    Parameters
    ----------
    expe_results : _type_
        _description_
    expe_indices : _type_
        _description_
    xp_causal_metrics : List[str]
        _description_
    candidate_params : List[str]
        _description_

    Returns
    -------
    _type_
        _description_
    """
    candidate_params = get_candidate_params(expe_results)
    measure_of_interest, expe_indices = get_expe_indices(
        expe_results, measure_of_interest=measure_of_interest
    )
    expe_rankings = get_metric_rankings_by_dataset(
        expe_results=expe_results,
        expe_indices=expe_indices,
        causal_metrics=expe_causal_metrics,
        candidate_params=candidate_params,
    )

    rankings_agg = get_rankings_aggregate(
        expe_rankings=expe_rankings,
        expe_indices=expe_indices,
        causal_metrics=expe_causal_metrics,
    )
    ax = plot_ranking_aggregation(
        rankings_aggregation=rankings_agg,
        expe_indices=expe_indices,
        x_metric_name=measure_of_interest,
        lowess_type="lowess_quantile",
        lowess_kwargs={"frac": 0.66, "it": 10, "quantile": quantile},
        show_legend=False,
        y_lim=ylim_ranking,
        reference_metric=reference_metric,
    )
    return ax
