"""Offline records -> independent checks -> statistics -> figures -> LaTeX.

No solver, generator, model client, credentials, or network calls are used.
仅使用保存记录生成当前论文图表。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.figure_style import COLORS, DISPLAY_NAMES, METHODS, STYLE, configure, export
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from scripts.case_study import case_evidence
from experiments.validate import recheck
from experiments.inference import clustered_sign_flip
from experiments.record_provenance import record_matches_manifest
from sar_alloc.instance_io import load_instance_from_json

METRICS = ("missed_priority", "unassigned_count", "energy_total")
PROTOCOL = json.loads((ROOT / 'configs/protocol.json').read_text(encoding='utf-8'))
SIZES = tuple(PROTOCOL['sizes'])
MATURITIES = tuple(f'M{round(f*100):02d}' for f in PROTOCOL['initial_fractions'])
SEEDS = tuple(PROTOCOL['instance_seeds']['test'])
TRIALS = tuple(range(0, 1201, 100))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_sources(root):
    lock = read(root / "results/integrity.json")
    for relative, digest in lock["files"].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"归档输入校验失败: {relative}")
    return len(lock["files"])


def compare(a, b, tolerance, service_only=False):
    for level, metric in enumerate(METRICS[:2] if service_only else METRICS, 1):
        if metric == "energy_total" and math.isclose(a[metric], b[metric], **tolerance):
            continue
        if a[metric] != b[metric]:
            return ("W" if a[metric] < b[metric] else "L"), level
    return "T", 0


def wtl(rows, key="outcome"):
    counts = Counter(row[key] for row in rows)
    return [counts[c] for c in "WTL"]


def load_verified(root, protocol):
    expected = {
        "status": "frozen", "sizes": list(SIZES), "initial_fractions": PROTOCOL['initial_fractions'],
        "trials": 1200, "action_trials": 100, "algorithm_seed": 0,
        "objective": list(METRICS), "checkpoints": list(TRIALS),
        "energy_definition": "transfer-only closed-route energy",
    }
    for key, value in expected.items():
        if protocol[key] != value:
            raise ValueError(f"Unexpected frozen protocol: {key}")
    if set(protocol["methods"]) != set(METHODS) or tuple(protocol["instance_seeds"]["test"]) != SEEDS:
        raise ValueError("Wrong method set or formal instance seeds")
    manifest = read(root / "data/test/manifest.json")
    starts = {(s["case_id"], s["maturity"]): s for s in read(root / "data/test/starts.json")}
    instances = {row["case_id"]: load_instance_from_json(root / row["instance_path"]) for row in manifest}
    if len(instances) != len(SIZES)*len(SEEDS) or len(starts) != len(SIZES)*len(SEEDS)*len(MATURITIES):
        raise ValueError("Unexpected dataset cardinality")
    # Independently verify the stored reference and starting routes, without search.
    for row in manifest:
        quality = recheck(instances[row["case_id"]], {"routes": row["reference_routes"], "unassigned": []})
        if (quality["missed_priority"], quality["unassigned_count"]) != (0, 0):
            raise ValueError("Reference route is not a complete feasible assignment")
    for key, start in starts.items():
        recheck(instances[key[0]], start["solution"])
    records, audit, pairs = [], [], defaultdict(dict)
    audited = {name: row["sha256"] for name, row in read(root / "results/provenance.json")["records"].items()}
    for path in sorted((root / "results/test/seed_0/records").glob("*.json")):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if audited.get(relative) != digest:
            raise ValueError(f"Record differs from integrity manifest: {relative}")
        record = read(path)
        key = (record["case_id"], record["maturity"])
        method = record["method"]
        if method not in METHODS or method in pairs[key]:
            raise ValueError(f"Unknown or duplicate method: {relative}")
        if record["status"] != "complete" or record["trials_used"] != 1200 or record["trials"] != 1200:
            raise ValueError(f"Incomplete record: {relative}")
        if record["algorithm_seed"] != 0 or record["objective_order"] != list(METRICS):
            raise ValueError(f"Wrong algorithm seed/objective: {relative}")
        if method == "agent":
            config = record["config"]
            if (not record_matches_manifest(record, path, root) or config["selectors"] != "random"
                    or config["feature_weights"] != 0 or len(record["actions"]) != 12
                    or len(record["decisions"]) != 12 or any(d["role"] != "step" for d in record["decisions"])):
                raise ValueError(f"Record does not match the fixed single-Agent configuration: {relative}")
        if record["n_tasks"] not in SIZES or record["instance_seed"] not in SEEDS or record["maturity"] not in MATURITIES:
            raise ValueError(f"Unexpected condition: {relative}")
        if tuple(p["trial"] for p in record["checkpoints"]) != TRIALS:
            raise ValueError(f"Unexpected checkpoint grid: {relative}")
        vectors = [tuple(p[m] for m in METRICS) for p in record["checkpoints"]]
        if any(after > before for before, after in zip(vectors, vectors[1:])):
            raise ValueError(f"Best-vector trajectory worsens: {relative}")
        if any((not math.isclose(record["checkpoints"][-1][m], record["final_quality"][m],
                                **protocol["analysis_plan"]["energy_reporting_tolerance"]))
               if m == "energy_total" else record["checkpoints"][-1][m] != record["final_quality"][m]
               for m in METRICS):
            raise ValueError(f"Final checkpoint mismatch: {relative}")
        for stored, solution in [(record["final_quality"], record["final_solution"]),
                                 (record["start_quality"], starts[key]["solution"])]:
            check = recheck(instances[key[0]], solution)
            for metric in METRICS:
                equal = math.isclose(check[metric], stored[metric], rel_tol=1e-10, abs_tol=1e-7)
                if not equal:
                    raise ValueError(f"Independent re-evaluation mismatch: {relative}/{metric}")
        final = recheck(instances[key[0]], record["final_solution"])
        audit.append({"record": relative, "sha256": digest, "passed": True,
                      "independent_quality": {m: final[m] for m in METRICS}})
        pairs[key][method] = record
        records.append(record)
    required = {(f"test_T{size}_s{seed}", maturity) for size in SIZES for seed in SEEDS for maturity in MATURITIES}
    if len(records) != len(required)*len(METHODS) or set(pairs) != required or any(set(pair) != set(METHODS) for pair in pairs.values()):
        raise ValueError("Formal conditions are missing or extra")
    tolerance = protocol["analysis_plan"]["energy_reporting_tolerance"]
    rows = []
    for (case_id, maturity), pair in pairs.items():
        a, b = pair["agent"], pair["basic"]
        if compare(a["start_quality"], b["start_quality"], tolerance)[0] != "T" or a["start_seed"] != b["start_seed"]:
            raise ValueError(f"Unpaired starting quality: {case_id}/{maturity}")
        aq, bq = a["final_quality"], b["final_quality"]
        outcome, level = compare(aq, bq, tolerance)
        service, _ = compare(aq, bq, tolerance, service_only=True)
        row = dict(case_id=case_id, maturity=maturity, size=a["n_tasks"], seed=a["instance_seed"],
                   initial_fraction=a["fraction"], outcome=outcome, level=level, service_outcome=service)
        for method in METHODS:
            row.update({f"{method}_L{i}": pair[method]["final_quality"][m] for i, m in enumerate(METRICS, 1)})
        row["same_service"] = service == "T"
        row["same_tasks"] = set(a["final_solution"]["unassigned"]) == set(b["final_solution"]["unassigned"])
        row["energy_reduction_pct"] = ((100 * (bq["energy_total"] - aq["energy_total"]) / bq["energy_total"])
                                        if service == "T" else None)
        rows.append(row)
    rows.sort(key=lambda r: (r["size"], r["seed"], MATURITIES.index(r["maturity"])))
    return records, rows, audit


def aggregate(rows):
    equal_service = [r for r in rows if r["same_service"]]
    data = {"conditions": len(rows), "wtl": wtl(rows), "service_wtl": wtl(rows, "service_outcome"),
            "same_service_conditions": len(equal_service),
            "mean_energy_reduction_pct": mean(r["energy_reduction_pct"] for r in equal_service) if equal_service else None}
    for method in METHODS:
        data[method] = {"mean_L1": mean(r[f"{method}_L1"] for r in rows),
                        "mean_L2": mean(r[f"{method}_L2"] for r in rows),
                        "mean_L3_all_samples": mean(r[f"{method}_L3"] for r in rows),
                        "mean_L3_equal_service": mean(r[f"{method}_L3"] for r in equal_service) if equal_service else None,
                        "complete": sum(r[f"{method}_L1"] == 0 and r[f"{method}_L2"] == 0 for r in rows)}
    return data


def summarize(records, rows, protocol):
    inference = clustered_sign_flip(rows)
    scores = inference['instance_scores']
    observed = inference['observed_sum']
    p = inference['p_two_sided']
    def energy(subset):
        return {"n": len(subset), "wtl": wtl(subset),
                "mean_reduction_pct": mean(r["energy_reduction_pct"] for r in subset) if subset else None,
                "median_reduction_pct": median(r["energy_reduction_pct"] for r in subset) if subset else None}
    equal = [r for r in rows if r["same_service"]]
    same = [r for r in equal if r["same_tasks"]]
    decisions = [d for r in records if r["method"] == "agent" for d in r["decisions"]]
    levels = Counter(f"{r['outcome']}_L{r['level']}" for r in rows)
    summary = {"records": len(records), "paired_conditions": len(rows), "base_instances": len(scores),
               "objective": list(METRICS), "overall": aggregate(rows),
               "by_size": {str(n): aggregate([r for r in rows if r["size"] == n]) for n in SIZES},
               "by_maturity": {m: aggregate([r for r in rows if r["maturity"] == m]) for m in MATURITIES},
               "first_decisive_layer": dict(sorted(levels.items())),
               "inference": {"unit": "base instance with dependent nested starts", "scores": scores,
                             "observed_sum": observed, "permutations": inference['permutations'], "p_two_sided": p,
                             "calculation": inference['calculation']},
               "energy_equal_service": energy(equal), "energy_same_tasks": energy(same),
               "energy_same_tasks_rule": "Equal L1/L2 and identical final unassigned task sets within the same instance/start; platform allocation and order may differ.",
               "energy_reporting_tolerance": protocol["analysis_plan"]["energy_reporting_tolerance"],
               "timings": {m: {"median_total_sec": median(r["total_time_sec"] for r in records if r["method"] == m),
                               "median_solver_sec": median(r["solver_time_sec"] for r in records if r["method"] == m)} for m in METHODS},
               "output_validation": {"decisions": len(decisions),
                    "local_envelope_repairs": sum(d["envelope_repair_count"] for d in decisions),
                    "repair_calls": sum(d["validation"]["repair_count"] for d in decisions),
                    "invalid_executed": sum(not d["validation"]["ok"] for d in decisions)},
               "llm_requests": sum(r["llm_usage"].get("requests", 0) for r in records),
               "limitations": ["One algorithm seed", "Only Basic as comparator", "No independent mechanism ablation",
                                "Concurrent descriptive timing", "Synthetic reference-feasible instances",
                                "Original fixed cohort reused for a same-condition regression review; not new independent evidence",
                                "Sequential development is not corrected by the single combined sign-flip p value"]}
    summary['by_cohort'] = {c['name']: {
        'overall': aggregate([r for r in rows if r['seed'] in c['seeds']]),
        'by_size': {str(n): aggregate([r for r in rows if r['seed'] in c['seeds'] and r['size']==n]) for n in SIZES}}
        for c in protocol.get('evaluation_cohorts', [{'name':'single_cohort','seeds':list(SEEDS)}])}
    summary['convergence']={str(n):{method:{metric:mean(point[metric] for record in records
        if record['n_tasks']==n and record['method']==method for point in record['checkpoints'] if point['trial']>0)
        for metric in METRICS[:2]} for method in METHODS} for n in SIZES}
    assert sum(levels.values()) == len(rows)
    assert sum(summary['overall']['wtl']) == len(rows)
    return summary


def number(value, digits=2):
    if value is None:
        return '---'
    return format(Decimal(str(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP), f".{digits}f")


def main_table(summary, out):
    equal_counts = [summary["by_size"][str(n)]["same_service_conditions"] for n in SIZES]
    size_labels = "/".join(f"T{n}" for n in SIZES)
    table_rows = []
    for group, stats in [(f"T{n}", summary["by_size"][str(n)]) for n in SIZES] + [("All", summary["overall"])]:
        row = {"group": group, "conditions": stats["conditions"],
               "equal_service_conditions": stats["same_service_conditions"],
               "mean_energy_reduction_pct": stats["mean_energy_reduction_pct"]}
        for method in METHODS:
            row.update({f"{method}_{key}": value for key, value in stats[method].items()})
            value = stats[method]["mean_L3_equal_service"]
            row[f"{method}_mean_L3_equal_service_scaled"] = None if value is None else value / 1000
        row["WTL"] = "/".join(map(str, stats["wtl"]))
        table_rows.append(row)
    write_csv(out / "data/main_table.csv", table_rows)
    lines = [r"% Generated by scripts/build_evidence.py; do not edit numbers.",
             r"\begin{table}[htbp]", r"\centering",
             r"\caption{Final task allocation performance and conditional transfer energy.}",
             r"\label{tab:main}",
             r"\sisetup{mode=text,detect-weight=true}", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}l *{4}{S[table-format=1.2]} *{2}{S[table-format=3.2]} c@{}}\toprule",
             r"Size & \multicolumn{2}{c}{Mean $L_1$} & \multicolumn{2}{c}{Mean $L_2$} & \multicolumn{2}{c}{Mean $L_3$ ($\times10^3$)$^{\dagger}$} & W/T/L \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & {Basic} & {IntALNS} & {Basic} & {IntALNS} & {Basic} & {IntALNS} & \\", r"\midrule"]
    for row in table_rows:
        if row["group"] == "All":
            lines.append(r"\midrule")
        cells = [row["group"]]
        for metric in ("mean_L1", "mean_L2"):
            values = [row[f"{method}_{metric}"] for method in METHODS]
            best = min(values)
            cells.extend((r"\bfseries " if value == best else "") + number(value) for value in values)
        energies = [row[f"{method}_mean_L3_equal_service_scaled"] for method in METHODS]
        best_energy = min((value for value in energies if value is not None), default=None)
        for energy in energies:
            cells.append(((r"\bfseries " if energy == best_energy else "") + number(energy))
                         if energy is not None else r"\multicolumn{1}{c}{---}")
        cells.append(row["WTL"])
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule\end{tabular}",
              r"\par\smallskip\begin{minipage}{\textwidth}\footnotesize",
              r"$\dagger$ Mean $L_3$ uses pairs with equal $L_1$ and $L_2$ ($n="
              + ",".join(map(str, equal_counts)) + f"$ for T50, T100, T300; {sum(equal_counts)} overall). Displayed energy values are divided by $10^3$. Bold marks lower means.",
              r"\end{minipage}", r"\end{table}"]
    (out / "tables/main_results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    notes = dict(zip(("SmallEqualServiceN", "MediumEqualServiceN", "LargeEqualServiceN"), equal_counts))
    for size, name in zip(SIZES, ("Small", "Medium", "Large")):
        stats = summary["by_size"][str(size)]
        notes[name + "EnergyReduction"] = number(stats["mean_energy_reduction_pct"])
        for method in METHODS:
            notes[method.title() + name + "Complete"] = stats[method]["complete"]
    (out / "tables/main_table_notes.tex").write_text(
        "% Generated full-sample completion and conditional energy statistics.\n" +
        "\n".join("\\newcommand{\\" + name + "}{" + str(value) + "}" for name, value in notes.items()) + "\n",
        encoding="utf-8", newline="\n")






def draw_convergence(curves, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(6.69, 2.48))
    fig.subplots_adjust(left=.09, right=.985, bottom=.14, top=.84, wspace=.29, hspace=.53)
    handles = []
    for row, metric in enumerate(METRICS[:2]):
        for column, size in enumerate(SIZES):
            ax = axes[row, column]
            upper = max(max(curves[size, method, metric][1:]) for method in METHODS) * 1.10
            for method in METHODS:
                line, = ax.plot(TRIALS[1:], curves[size, method, metric][1:], **STYLE[method],
                                linewidth=1.25, markersize=3.0, markerfacecolor="white",
                                markeredgewidth=.8, zorder=3)
                if row == column == 0:
                    handles.append(line)
            panel = "abcdef"[row * len(SIZES) + column]
            ax.set(title=f"({panel}) T{size}", xlim=(70, 1230), ylim=(0, upper))
            if row == 1:
                ax.set_xlabel("iters", labelpad=2)
            ax.set_xticks((100, 400, 800, 1200))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
            ax.tick_params(length=2.5, width=.6, labelsize=8, pad=2)
            ax.grid(axis="y", color="#e4e4e4", linewidth=.45, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Mean priority\nloss", labelpad=5)
    axes[1, 0].set_ylabel("Mean unassigned\ncount", labelpad=5)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.535, 1.015),
               ncol=2, frameon=False, fontsize=9, handlelength=2.7, columnspacing=2.2)
    export(fig, Path(output_dir) / "convergence")


def macros(summary, out):
    overall = summary["overall"]
    data = {"AllWins": overall["wtl"][0], "AllTies": overall["wtl"][1], "AllLosses": overall["wtl"][2],
            "PairCount": summary['paired_conditions'], "InstanceCount": summary['base_instances'],
            "EqualServiceN": summary['energy_equal_service']['n'],
            "EnergyWins": summary['energy_equal_service']['wtl'][0], "EnergyTies": summary['energy_equal_service']['wtl'][1], "EnergyLosses": summary['energy_equal_service']['wtl'][2],
            "EnergyMedian": number(summary['energy_equal_service']['median_reduction_pct']),
            "PriorityWins": summary['first_decisive_layer'].get('W_L1',0),
            "ServiceWins": overall["service_wtl"][0], "ServiceTies": overall["service_wtl"][1], "ServiceLosses": overall["service_wtl"][2],
            "AgentComplete": overall["agent"]["complete"], "BasicComplete": overall["basic"]["complete"],
            "ExactP": summary["inference"]["p_two_sided"],
            "PriorityReduction": number(100 * (1 - overall["agent"]["mean_L1"] / overall["basic"]["mean_L1"])),
            "CountReduction": number(100 * (1 - overall["agent"]["mean_L2"] / overall["basic"]["mean_L2"])),
            "EnergyMean": number(summary["energy_equal_service"]["mean_reduction_pct"]),
            "EnergyChangeDirection": "reduction" if summary["energy_equal_service"]["mean_reduction_pct"] >= 0 else "increase",
            "EnergyChangeMagnitude": number(abs(summary["energy_equal_service"]["mean_reduction_pct"])),
            "SameTasksN": summary["energy_same_tasks"]["n"],
            "SameTasksMean": number(summary["energy_same_tasks"]["mean_reduction_pct"]),
            "AgentSeconds": number(summary["timings"]["agent"]["median_total_sec"]),
            "BasicSeconds": number(summary["timings"]["basic"]["median_total_sec"])}
    for method in METHODS:
        data[method.title() + "PriorityMean"] = number(overall[method]["mean_L1"])
        data[method.title() + "CountMean"] = number(overall[method]["mean_L2"])
    for size,name in zip(SIZES,('Small','Medium','Large')):
        stats = summary['by_size'][str(size)]
        for key,value in zip(('Wins','Ties','Losses'),stats['wtl']):
            data[name+key] = value
        for method in METHODS:
            for metric,label in [('mean_L1','PriorityMean'),('mean_L2','CountMean')]:
                data[method.title()+name+label] = number(stats[method][metric])
            conditional_energy = stats[method]['mean_L3_equal_service']
            data[method.title()+name+'ConditionalEnergyMean'] = number(
                None if conditional_energy is None else conditional_energy / 1000)
    text = "% Generated evidence numbers; no hand transcription.\n" + "\n".join(
        "\\newcommand{\\" + key + "}{" + str(value) + "}" for key, value in data.items()) + "\n"
    (out / "tables/numbers.tex").write_text(text, encoding="utf-8", newline="\n")




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--table-only", action="store_true", help="校验归档记录并生成主结果表及数值宏")
    args = parser.parse_args()
    root = args.project.resolve()
    out = (args.out or root / "paper").resolve()
    if out in (root, root / "results", root / "data", root / "configs") or any(p in out.parents for p in (root / "results", root / "data", root / "configs")):
        raise ValueError("Evidence output must not overwrite frozen inputs")
    for directory in ("data", "figures", "tables"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    protected_count = verify_sources(root)
    protocol = read(root / "configs/protocol.json")
    records, rows, audit = load_verified(root, protocol)
    summary = summarize(records, rows, protocol)
    previous = read(root / "results/reference_summary.json")
    if summary["overall"]["wtl"] != previous["overall"]["wtl"] or summary["inference"]["p_two_sided"] != previous["inference"]["p_two_sided"]:
        raise ValueError("Recomputed results disagree with original report")
    if summary['convergence'] != previous['convergence']:
        raise ValueError('Recomputed real-checkpoint means disagree with the formal convergence assessment')
    if args.table_only:
        main_table(summary, out)
        macros(summary, out)
        verify_sources(root)
        print(f"已从 {len(records)} 份记录生成主结果表；条件能耗均值使用 {summary['overall']['same_service_conditions']} 对同服务水平结果。")
        return
    write_csv(out / "data/paired_results_verified.csv", rows)
    write_json(out / "data/independent_audit.json", {"passed": True, "protected_files": protected_count,
               "audited_solutions": len(audit), "reference_routes_checked": summary['base_instances'], "starts_checked": len(rows), "records": audit})
    write_json(out / "data/analysis_summary.json", summary)
    configure()
    main_table(summary, out)
    macros(summary, out)
    case_evidence(records, out)
    curves, curve_rows = {}, []
    for size in SIZES:
        for method in METHODS:
            subset = [r for r in records if r["n_tasks"] == size and r["method"] == method]
            for metric in METRICS[:2]:
                curves[size, method, metric] = [mean(r["checkpoints"][j][metric] for r in subset) for j in range(len(TRIALS))]
            for j, trial in enumerate(TRIALS):
                curve_rows.append(dict(n_tasks=size, method=method, trial=trial,
                    mean_missed_priority=curves[size, method, METRICS[0]][j],
                    mean_unassigned_count=curves[size, method, METRICS[1]][j], conditions=len(subset)))
    write_csv(out / "data/convergence_means.csv", curve_rows)
    draw_convergence(curves, output_dir=out / "figures")
    # Existing formal reports are retained unchanged; publication output lives here.
    verify_sources(root)
    print(f"Offline build passed: {len(records)} audited records, {len(rows)} pairs; W/T/L={summary['overall']['wtl']}; p={summary['inference']['p_two_sided']}")


if __name__ == "__main__":
    main()
