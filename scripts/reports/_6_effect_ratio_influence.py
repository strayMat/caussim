import re
from typing import List, Tuple
import pandas as pd
import numpy as np

import logging
import pytest
from caussim.reports import (
    plot_agreement_w_tau_risk,
    get_best_estimator_by_dataset,
    plot_evaluation_metric,
    read_logs,
    save_figure_to_folders,
    get_nuisances_type,
    get_candidate_params,
    get_expe_indices,
)
from caussim.reports.plots_utils import (
    CAUSAL_METRIC_LABELS,
    CAUSAL_METRICS,
    CAUSSIM_LABEL,
    DATASETS_PALETTE,
    EVALUATION_METRIC_LABELS,
    METRIC_LABEL,
    METRIC_ORDER,
    EFFECT_RATIO_BIN_COL,
    EFFECT_RATIO_BIN_PALETTE,
    get_kendall_by_effect_ratio_bin,
    plot_kendall_compare_vs_measure,
    plot_metric_legend,
    DATASET_LABEL,
    EFFECT_RATIO_BIN_LABELS,
)
from caussim.utils import *
from caussim.config import *

sns.set_style("whitegrid")

script_name = "_6_effect_ratio_influence"


DATASET_EXPERIMENTS = [
    (
        {
            CAUSSIM_LABEL: Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_35-139.parquet"
            )
        },
        None,
        True,
        True,
        (-0.5, 1.05),
        "effect_ratio"
    ),
    (
        {
            CAUSSIM_LABEL: Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_35-139.parquet"
            )
        },
        None,
        True,
        True,
        (-0.5, 1.05),
        "effect_ratio_sym"
    ),
    (
        {
            CAUSSIM_LABEL: Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_35-139.parquet"
            )
        },
        None,
        True,
        True,
        (-0.5, 1.05),
        "effect_ratio_sym2"
    ),
    (
        {
            CAUSSIM_LABEL: Path(
                DIR2EXPES
                / "caussim_save"
                / "caussim__nuisance_non_linear__candidates_ridge__overlap_35-139.parquet"
            )
        },
        None,
        True,
        True,
        (-0.5, 1.05),
        "effect_variation"
    ),
]

dataset_label = "Dataset"

@pytest.mark.parametrize(
    "xp_paths, reference_metric, plot_legend, plot_middle_bin, xlim, effect_ratio_measure_variant",
    DATASET_EXPERIMENTS,
)
def test_plot_effect_ratio_difference(
    xp_paths: Dict[str, Path],
    reference_metric: str,
    plot_legend: bool,
    plot_middle_bin: bool,
    xlim: Tuple[float, float],
    effect_ratio_measure_variant: str
):
    all_expe_results_by_bin = []
    for expe_name, xp_path in xp_paths.items():
        # for all datasets
        xp_res_, _ = read_logs(xp_path)
        expe_causal_metrics = [
            metric for metric in CAUSAL_METRICS if (metric in xp_res_.columns)
        ]
        dataset_name = xp_res_["dataset_name"].values[0]
        
        kendall_by_effect_ratio_bin, evaluation_metric = (
            get_kendall_by_effect_ratio_bin(
                xp_res=xp_res_,
                reference_metric=reference_metric,
                expe_causal_metrics=expe_causal_metrics,
                plot_middle_overlap_bin=plot_middle_bin,
                measure_of_interest=effect_ratio_measure_variant,
            )
        )
        kendall_by_effect_ratio_bin[DATASET_LABEL] = expe_name
        all_expe_results_by_bin.append(kendall_by_effect_ratio_bin)
    all_expe_results_by_bin_df = pd.concat(all_expe_results_by_bin)
    ds_used = all_expe_results_by_bin_df[DATASET_LABEL].unique()
    ds_order = [
        ds_label_ for ds_label_ in DATASETS_PALETTE.keys() if ds_label_ in ds_used
    ]
    # Weak overlap first
    effect_ratio_order = [
        ov_
        for ov_ in EFFECT_RATIO_BIN_LABELS
        if ov_ in all_expe_results_by_bin_df[EFFECT_RATIO_BIN_COL[effect_ratio_measure_variant]].unique()
    ]  # [::-1]

    metric_order = [
        CAUSAL_METRIC_LABELS[metric_label_] for metric_label_ in METRIC_ORDER
    ]

    effect_ratio_measure_distrib = (
        all_expe_results_by_bin_df[effect_ratio_measure_variant]
        .drop_duplicates()
        .describe(percentiles=[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    )
    print(
        f"Effect ratio measure distribution: {effect_ratio_measure_distrib.transpose().to_markdown()}"
    )
    # Full figure
    g = sns.FacetGrid(
        data=all_expe_results_by_bin_df,
        col=DATASET_LABEL,
        col_wrap=2,
        col_order=ds_order,
        height=10,
        aspect=1,
    )
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=metric_order,
        hue=EFFECT_RATIO_BIN_COL[effect_ratio_measure_variant],
        hue_order=effect_ratio_order,
        palette=EFFECT_RATIO_BIN_PALETTE,
        linewidth=2,
        notch=True,
        medianprops={"linewidth": 6},
    )
    if f"{evaluation_metric}_long" in EVALUATION_METRIC_LABELS.keys():
        suptitle = EVALUATION_METRIC_LABELS[f"{evaluation_metric}_long"]
    else:
        suptitle = EVALUATION_METRIC_LABELS[evaluation_metric]
    g.set(xlabel=suptitle, ylabel="", xlim=xlim)
    if reference_metric is not None:
        ref_metric_str = f"_ref_metric_{reference_metric}"
    else:
        ref_metric_str = "kendall"
    if plot_legend:
        g.add_legend()
        legend_data = g._legend_data  # type: ignore
        g._legend.remove()  # type: ignore
        g.add_legend(
            title=EFFECT_RATIO_BIN_COL[effect_ratio_measure_variant],
            legend_data=legend_data,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.25, 1.15),
            columnspacing=0.5,
            prop={"size": 36},
        )
    else:
        g._legend.remove()  # type: ignore
    save_figure_to_folders(
        figure_name=Path(
            script_name
            + f"/effect_ratio_by_bin_comparaison_{ref_metric_str}_by_{DATASET_LABEL}_{effect_ratio_measure_variant}"
        ),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )

    # R_risk only figure
    sub_metrics = [
        CAUSAL_METRIC_LABELS["r_risk"],
        CAUSAL_METRIC_LABELS["oracle_r_risk"],
    ]
    r_risk_data = all_expe_results_by_bin_df[
        all_expe_results_by_bin_df[METRIC_LABEL].isin(sub_metrics)
    ]
    g = sns.FacetGrid(
        data=r_risk_data,
        col=DATASET_LABEL,
        col_order=ds_order,
        height=6,
        aspect=1.8,
        col_wrap=2,
    )
    g = g.map_dataframe(
        sns.boxplot,
        x=evaluation_metric,
        y=METRIC_LABEL,
        order=sub_metrics,
        hue=EFFECT_RATIO_BIN_COL[effect_ratio_measure_variant],
        hue_order=effect_ratio_order,
        palette=EFFECT_RATIO_BIN_PALETTE,
        linewidth=2,
        notch=True,
        medianprops={"linewidth": 6},
    )
    g.set(xlabel=suptitle, ylabel="", xlim=(0, 1.05))
    g.add_legend(
        legend_data=legend_data,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.35, 1.05),
        columnspacing=0.5,
        prop={"size": 36},
    )

    save_figure_to_folders(
        figure_name=Path(
            script_name
            + f"/effect_ratio_by_bin_comparaison_{ref_metric_str}_by_{DATASET_LABEL}_r_risk_only_{effect_ratio_measure_variant}"
        ),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )

    return


def test_plot_metrics_legend():
    ncol = 2
    plot_metric_legend(CAUSAL_METRICS, ncol=ncol)
    save_figure_to_folders(
        figure_name=Path(f"legend_metrics_ncol={ncol}"),
        figure_dir=True,
        notes_dir=False,
        paper_dir=True,
    )
