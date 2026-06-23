from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


@dataclass
class ReportingArtifacts:
    dataset_table_tex: Path
    main_table_tex: Path
    pairwise_table_tex: Path
    operator_stats_tex: Path
    rank_table_csv: Path
    anytime_plot: Path
    complexity_plot: Path
    cost_validation_plot: Path
    rank_plot: Path
    manifest: Path


@dataclass
class PaperDraftArtifacts:
    filled_tex: Path
    manifest: Path


@dataclass
class LatexBuildArtifacts:
    pdf_path: Path | None
    log_path: Path | None
    success: bool


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _method_label(name: str) -> str:
    mapping = {
        "proposed": "Proposed",
        "linear_ea": "Linear-EA",
        "graph_serialcost": "SerialCost",
        "graph_nostaged": "NoStaged",
        "random_graph": "Random",
    }
    return mapping.get(name, name)


def _method_order(frame: pd.DataFrame | None = None) -> list[str]:
    preferred = ["proposed", "linear_ea", "graph_serialcost", "graph_nostaged", "random_graph"]
    if frame is None or frame.empty or "method" not in frame.columns:
        return preferred
    present = [method for method in preferred if method in set(frame["method"])]
    extras = sorted(set(frame["method"]) - set(present))
    return present + extras


def _replace_labelled_environment(tex: str, label: str, replacement: str) -> str:
    label_token = f"\\label{{{label}}}"
    label_index = tex.find(label_token)
    if label_index < 0:
        return tex

    table_begin = tex.rfind("\\begin{table}[t]", 0, label_index)
    figure_begin = tex.rfind("\\begin{figure}[t]", 0, label_index)
    begin_index = max(table_begin, figure_begin)
    if begin_index < 0:
        return tex

    if begin_index == table_begin:
        end_token = "\\end{table}"
    else:
        end_token = "\\end{figure}"
    end_index = tex.find(end_token, label_index)
    if end_index < 0:
        return tex
    end_index += len(end_token)
    return tex[:begin_index] + replacement + tex[end_index:]


def build_dataset_table_tex(df: pd.DataFrame) -> str:
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Dataset & $n$ & $d$ & $d_{\\mathrm{num}}$ & $d_{\\mathrm{cat}}$ & Miss (\\%) & $C$ \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{_latex_escape(row['dataset'])} & {int(row['n_samples'])} & {int(row['n_features'])} & {int(row['n_numeric'])} & {int(row['n_categorical'])} & {100.0 * float(row['missing_fraction']):.1f} & {int(row['n_classes'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_main_table_tex(summary: pd.DataFrame, pairwise: pd.DataFrame) -> str:
    method_order = _method_order(summary)
    datasets = list(summary["dataset"].unique())
    significance = {row["baseline"]: row["p_value"] for _, row in pairwise.iterrows()}
    lines = [
        "\\begin{tabular}{l" + "c" * len(method_order) + "}",
        "\\toprule",
        "Dataset & " + " & ".join(f"\\texttt{{{_method_label(method)}}}" for method in method_order) + " \\\\",
        "\\midrule",
    ]
    avg_ranks: dict[str, float] = {}
    pivot = summary.pivot(index="dataset", columns="method", values="mean_score")
    if not pivot.empty:
        avg_ranks = pivot.rank(axis=1, ascending=False, method="average").mean(axis=0).to_dict()
    for dataset in datasets:
        subset = summary[summary["dataset"] == dataset].set_index("method")
        best = subset["mean_score"].max()
        cells = [_latex_escape(dataset)]
        for method in method_order:
            if method not in subset.index:
                cells.append("---")
                continue
            row = subset.loc[method]
            std = 0.0 if pd.isna(row["std_score"]) else float(row["std_score"])
            value = f"{float(row['mean_score']):.3f}$\\pm${std:.3f}"
            if abs(float(row["mean_score"]) - float(best)) < 1e-12:
                value = f"\\textbf{{{value}}}"
            if method == "proposed" and significance.get("linear_ea") is not None and significance["linear_ea"] < 0.05:
                value += "$^{\\dagger}$"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    if avg_ranks:
        lines.append("\\midrule")
        lines.append("Avg. rank & " + " & ".join(f"{avg_ranks.get(method, np.nan):.3f}" for method in method_order) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_pairwise_table_tex(records: pd.DataFrame) -> str:
    method_order = _method_order(records)
    pivot = records.pivot_table(index=["dataset", "seed"], columns="method", values="score")
    lines = [
        "\\begin{tabular}{l" + "c" * len(method_order) + "}",
        "\\toprule",
        " & " + " & ".join(_method_label(method) for method in method_order) + " \\\\",
        "\\midrule",
    ]
    for row_method in method_order:
        cells = [_method_label(row_method)]
        for col_method in method_order:
            if row_method == col_method:
                cells.append("---")
                continue
            if row_method not in pivot.columns or col_method not in pivot.columns:
                cells.append("---")
                continue
            paired = pivot[[row_method, col_method]].dropna()
            if paired.empty:
                cells.append("---")
                continue
            delta = paired[row_method] - paired[col_method]
            if delta.eq(0).all():
                p_value = 1.0
            else:
                p_value = float(wilcoxon(delta).pvalue)
            value = f"{p_value:.4f}"
            if p_value < 0.05:
                value = f"\\textbf{{{value}}}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_operator_stats_tex(history: pd.DataFrame) -> str:
    if history.empty or "operator" not in history.columns:
        return ""
    df = history.copy()
    df["accepted"] = df["status"].isin(["ok", "screened_out"])
    df["improved_incumbent"] = df["improved_incumbent"].fillna(False)
    summary = (
        df.groupby("operator", as_index=False)
        .agg(
            frequency=("operator", "count"),
            accept_rate=("accepted", "mean"),
            improvement_rate=("improved_incumbent", "mean"),
        )
        .sort_values("frequency", ascending=False)
    )
    summary["frequency_pct"] = 100.0 * summary["frequency"] / max(1, summary["frequency"].sum())
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Operator & Frequency (\\%) & Accept Rate (\\%) & Improvement Rate (\\%) \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{_latex_escape(str(row['operator']))} & {float(row['frequency_pct']):.1f} & {100.0 * float(row['accept_rate']):.1f} & {100.0 * float(row['improvement_rate']):.1f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _save_rank_plot(summary: pd.DataFrame, out_path: Path) -> Path:
    pivot = summary.pivot(index="dataset", columns="method", values="mean_score")
    ranks = pivot.rank(axis=1, ascending=False, method="average").mean(axis=0).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.scatter(ranks.values, np.zeros_like(ranks.values), s=90)
    for method, rank in ranks.items():
        ax.text(rank, 0.05, _method_label(method), ha="center", va="bottom", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title("Average method ranks across datasets")
    ax.set_xlim(max(0.5, float(ranks.min()) - 0.5), float(ranks.max()) + 0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_anytime_plot(history: pd.DataFrame, out_path: Path) -> Path:
    df = history.dropna(subset=["elapsed_seconds", "best_so_far"]).copy()
    if df.empty:
        return out_path
    df["time_bin"] = df["elapsed_seconds"].round(0)
    grouped = df.groupby(["method", "time_bin"], as_index=False)["best_so_far"].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for method, sub in grouped.groupby("method"):
        ax.plot(
            sub["time_bin"],
            sub["best_so_far"],
            marker="o",
            linewidth=1.5,
            markersize=3,
            label=_method_label(method),
        )
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Mean incumbent balanced accuracy")
    ax.set_title("Anytime convergence")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_complexity_plot(history: pd.DataFrame, out_path: Path) -> Path:
    df = history.dropna(subset=["generation"]).copy()
    if df.empty:
        return out_path
    grouped = df.groupby(["method", "generation"], as_index=False).agg(
        mean_nodes=("nodes", "mean"),
        mean_branches=("branches", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method, sub in grouped.groupby("method"):
        axes[0].plot(sub["generation"], sub["mean_nodes"], label=_method_label(method))
        axes[1].plot(sub["generation"], sub["mean_branches"], label=_method_label(method))
    axes[0].set_title("Preprocessing nodes")
    axes[1].set_title("Branch count")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _save_cost_validation_plot(records: pd.DataFrame, out_path: Path) -> Path:
    df = records.dropna(subset=["estimated_cost", "runtime_seconds"]).copy()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(df["estimated_cost"], df["runtime_seconds"], s=18, alpha=0.65)
    ax.set_xlabel("Predicted cost")
    ax.set_ylabel("Actual runtime (s)")
    ax.set_title("Cost model validation")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _wrap_table_input(input_name: str, caption: str, label: str, size_cmd: str = "\\small") -> str:
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            size_cmd,
            f"\\input{{{input_name}}}",
            "\\end{table}",
        ]
    )


def _wrap_figure(path_name: str, caption: str, label: str, width: str = "0.95\\linewidth") -> str:
    return "\n".join(
        [
            "\\begin{figure}[t]",
            "\\centering",
            f"\\includegraphics[width={width}]{{{path_name}}}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{figure}",
        ]
    )


def _replace_todo_count(tex: str, placeholder: str, value: str) -> str:
    return tex.replace(f"\\todo{{{placeholder}}}", value)


def generate_artifacts(output_dir: str | Path) -> ReportingArtifacts:
    base = Path(output_dir)
    records = pd.read_csv(base / "experiment_records.csv")
    summary = pd.read_csv(base / "experiment_summary.csv")
    pairwise = pd.read_csv(base / "pairwise_tests.csv")
    history = pd.read_csv(base / "search_history.csv") if (base / "search_history.csv").exists() else pd.DataFrame()
    datasets = pd.read_csv(base / "dataset_characteristics.csv")

    dataset_table_tex = base / "paper_dataset_table.tex"
    main_table_tex = base / "paper_main_table.tex"
    pairwise_table_tex = base / "paper_pairwise_table.tex"
    operator_stats_tex = base / "paper_operator_stats.tex"
    rank_table_csv = base / "average_ranks.csv"
    anytime_plot = base / "figure_anytime.png"
    complexity_plot = base / "figure_complexity.png"
    cost_validation_plot = base / "figure_cost_validation.png"
    rank_plot = base / "figure_cd_like_ranks.png"
    manifest = base / "reporting_manifest.json"

    dataset_table_tex.write_text(build_dataset_table_tex(datasets), encoding="utf-8")
    main_table_tex.write_text(build_main_table_tex(summary, pairwise), encoding="utf-8")
    pairwise_table_tex.write_text(build_pairwise_table_tex(records), encoding="utf-8")
    operator_stats_tex.write_text(build_operator_stats_tex(history), encoding="utf-8")

    rank_pivot = summary.pivot(index="dataset", columns="method", values="mean_score")
    ranks = rank_pivot.rank(axis=1, ascending=False, method="average").mean(axis=0).reset_index()
    ranks.columns = ["method", "avg_rank"]
    ranks.to_csv(rank_table_csv, index=False)

    _save_anytime_plot(history, anytime_plot)
    _save_complexity_plot(history, complexity_plot)
    _save_cost_validation_plot(records, cost_validation_plot)
    _save_rank_plot(summary, rank_plot)

    manifest.write_text(
        json.dumps(
            {
                "dataset_table_tex": str(dataset_table_tex.resolve()),
                "main_table_tex": str(main_table_tex.resolve()),
                "pairwise_table_tex": str(pairwise_table_tex.resolve()),
                "operator_stats_tex": str(operator_stats_tex.resolve()),
                "rank_table_csv": str(rank_table_csv.resolve()),
                "anytime_plot": str(anytime_plot.resolve()),
                "complexity_plot": str(complexity_plot.resolve()),
                "cost_validation_plot": str(cost_validation_plot.resolve()),
                "rank_plot": str(rank_plot.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ReportingArtifacts(
        dataset_table_tex=dataset_table_tex.resolve(),
        main_table_tex=main_table_tex.resolve(),
        pairwise_table_tex=pairwise_table_tex.resolve(),
        operator_stats_tex=operator_stats_tex.resolve(),
        rank_table_csv=rank_table_csv.resolve(),
        anytime_plot=anytime_plot.resolve(),
        complexity_plot=complexity_plot.resolve(),
        cost_validation_plot=cost_validation_plot.resolve(),
        rank_plot=rank_plot.resolve(),
        manifest=manifest.resolve(),
    )


def assemble_draft(
    draft_path: str | Path,
    output_dir: str | Path,
    assembled_tex_path: str | Path | None = None,
) -> PaperDraftArtifacts:
    draft = Path(draft_path)
    results_dir = Path(output_dir)
    assembled_tex = Path(assembled_tex_path) if assembled_tex_path is not None else Path("paper_filled.tex")
    artifacts = generate_artifacts(results_dir)

    draft_text = draft.read_text(encoding="utf-8")
    relative_results = Path(os.path.relpath(results_dir.resolve(), assembled_tex.resolve().parent.resolve()))

    dataset_count = len(pd.read_csv(results_dir / "dataset_characteristics.csv"))
    records = pd.read_csv(results_dir / "experiment_records.csv")
    seeds = int(records["seed"].nunique()) if not records.empty and "seed" in records.columns else 0
    rho = spearmanr(records["estimated_cost"], records["runtime_seconds"]).statistic if not records.empty else np.nan
    rho_text = "N/A" if pd.isna(rho) else f"{float(rho):.2f}"

    draft_text = _replace_todo_count(draft_text, "N", str(dataset_count))
    draft_text = _replace_todo_count(draft_text, "30", str(seeds))
    draft_text = _replace_todo_count(draft_text, "600", "600")
    draft_text = _replace_todo_count(draft_text, "0.XX", rho_text)

    draft_text = _replace_labelled_environment(
        draft_text,
        "tab:datasets",
        _wrap_table_input(
            f"{relative_results.as_posix()}/paper_dataset_table.tex",
            "Dataset characteristics. $n$: samples, $d$: features, $d_{\\mathrm{num}}$/$d_{\\mathrm{cat}}$: numerical/categorical feature counts, Miss: fraction of missing values, $C$: number of classes.",
            "tab:datasets",
        ),
    )
    draft_text = _replace_labelled_environment(
        draft_text,
        "tab:main",
        _wrap_table_input(
            f"{relative_results.as_posix()}/paper_main_table.tex",
            f"Mean balanced accuracy ($\\pm$ std) over {seeds} runs. Best mean per dataset in \\textbf{{bold}}. $\\dagger$: significantly better than \\texttt{{Linear-EA}} at $p<0.05$ (Wilcoxon signed-rank).",
            "tab:main",
        ),
    )
    draft_text = _replace_labelled_environment(
        draft_text,
        "tab:operators:stats",
        _wrap_table_input(
            f"{relative_results.as_posix()}/paper_operator_stats.tex",
            "Variation operator statistics aggregated over all evolutionary methods and datasets. ``Accept rate'': fraction of offspring from this operator that entered the population. ``Improvement rate'': fraction that improved the incumbent.",
            "tab:operators:stats",
        ),
    )
    draft_text = _replace_labelled_environment(
        draft_text,
        "tab:pairwise",
        _wrap_table_input(
            f"{relative_results.as_posix()}/paper_pairwise_table.tex",
            "Pairwise Wilcoxon signed-rank test $p$-values. Each cell reports the $p$-value for the row method vs. the column method across all paired runs. Significant results ($p<0.05$) are shown in \\textbf{bold}.",
            "tab:pairwise",
        ),
    )

    draft_text = _replace_labelled_environment(
        draft_text,
        "fig:cd",
        _wrap_figure(
            f"{relative_results.as_posix()}/figure_cd_like_ranks.png",
            f"Critical difference style rank visualization over {dataset_count} datasets. Lower rank indicates better average performance.",
            "fig:cd",
        ),
    )
    draft_text = _replace_labelled_environment(
        draft_text,
        "fig:cost-validation",
        _wrap_figure(
            f"{relative_results.as_posix()}/figure_cost_validation.png",
            f"Cost model validation: predicted vs. actual evaluation time. Spearman $\\rho={rho_text}$.",
            "fig:cost-validation",
        ),
    )
    draft_text = _replace_labelled_environment(
        draft_text,
        "fig:anytime",
        _wrap_figure(
            f"{relative_results.as_posix()}/figure_anytime.png",
            f"Anytime convergence: mean incumbent balanced accuracy over wall-clock time, averaged over {seeds} runs.",
            "fig:anytime",
        ),
    )
    draft_text = _replace_labelled_environment(
        draft_text,
        "fig:complexity",
        _wrap_figure(
            f"{relative_results.as_posix()}/figure_complexity.png",
            "Evolution of graph complexity metrics over generations.",
            "fig:complexity",
        ),
    )

    if "\\label{fig:bestgraphs}" in draft_text:
        draft_text = _replace_labelled_environment(
            draft_text,
            "fig:bestgraphs",
            "\n".join(
                [
                    "\\begin{figure}[t]",
                    "\\centering",
                    "\\fbox{\\parbox{0.92\\linewidth}{\\centering",
                    "Graph visualisation was not auto-generated by the current benchmark pipeline. ",
                    "This placeholder remains so representative best-found DAGs can be added after selecting specific datasets and exporting the corresponding structures.}}",
                    "\\caption{Best preprocessing graphs found by \\texttt{Proposed} on representative datasets.}",
                    "\\label{fig:bestgraphs}",
                    "\\end{figure}",
                ]
            ),
        )

    assembled_tex.parent.mkdir(parents=True, exist_ok=True)
    assembled_tex.write_text(draft_text, encoding="utf-8")
    manifest = assembled_tex.with_name(f"{assembled_tex.stem}_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "draft_path": str(draft.resolve()),
                "output_dir": str(results_dir.resolve()),
                "assembled_tex": str(assembled_tex.resolve()),
                "reporting_manifest": str(artifacts.manifest),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PaperDraftArtifacts(filled_tex=assembled_tex.resolve(), manifest=manifest.resolve())


def compile_latex_document(tex_path: str | Path, runs: int = 2) -> LatexBuildArtifacts:
    tex_file = Path(tex_path).resolve()
    workdir = tex_file.parent
    pdf_path = tex_file.with_suffix(".pdf")
    log_path = tex_file.with_suffix(".build.log")
    command = [
        "pdflatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_file.name,
    ]
    outputs: list[str] = []
    success = True
    for _ in range(max(1, runs)):
        result = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        outputs.append(result.stdout)
        outputs.append(result.stderr)
        if result.returncode != 0:
            success = False
            break
    log_path.write_text("\n".join(outputs), encoding="utf-8")
    return LatexBuildArtifacts(
        pdf_path=pdf_path if success and pdf_path.exists() else None,
        log_path=log_path if log_path.exists() else None,
        success=success and pdf_path.exists(),
    )
