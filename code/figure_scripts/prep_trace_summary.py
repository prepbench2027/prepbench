#!/usr/bin/env python3
"""
prep_trace_summary.py  --  shrink a huge candidate-level search history into a
few-KB summary that is sufficient to (a) close the score/full-eval attribution,
(b) measure staged-evaluation behaviour, and (c) cross-check against the
run-level experiment_records.

Streams in chunks (multi-gigabyte input is fine), reads .csv or .gz, and accepts
one combined file or several per-budget files. Robust to rows with missing
seed/budget, and can derive the budget from a folder name like
benchmark_results300/ when the budget column is absent or empty.

USAGE
    python prep_trace_summary.py search_history.csv --out trace_summary
    python prep_trace_summary.py r60/search_history.csv r300/search_history.csv r600/search_history.csv --out trace_summary
    python prep_trace_summary.py search_history.csv.gz --out trace_summary

OUTPUT (all tiny -- zip the folder and upload it)
    trace_summary/status_summary.csv      distinct status values x config x budget
    trace_summary/per_run_incumbent.csv   one row per (dataset,config,seed,budget)
    trace_summary/staged_metrics.csv       per (config,budget): stage1<->full corr,
                                           promotion rate, false-negative rate
    trace_summary/stage_full_sample.csv    capped random (stage1,score) pairs
"""
import argparse
import math
import os
import re
import random
from collections import defaultdict, Counter

import pandas as pd
from scipy.stats import spearmanr

random.seed(0)

WANT = {
    "dataset": ["dataset"],
    "config": ["config", "configuration"],
    "seed": ["seed"],
    "budget": ["budget"],
    "status": ["status", "state", "outcome"],
    "score": ["score", "full_score", "final_score"],
    "stage1_score": ["stage1_score", "stage1", "screen_score", "subsample_score"],
    "best_so_far": ["best_so_far", "best", "incumbent_score"],
    "improved_incumbent": ["improved_incumbent", "is_incumbent", "improved"],
}
RESERVOIR_CAP = 4000
CHUNK = 200_000


def resolve_columns(path):
    head = pd.read_csv(path, nrows=0)
    have = {c.strip().lower(): c for c in head.columns}
    mapping = {}
    for logical, spellings in WANT.items():
        for s in spellings:
            if s in have:
                mapping[logical] = have[s]
                break
    # budget is NOT required here: it can be derived from the file path.
    required = [k for k in ("dataset", "config", "seed", "status")
                if k not in mapping]
    if required:
        raise SystemExit(
            f"ERROR: could not find required column(s) {required} in {path}. "
            f"Found headers: {list(head.columns)}")
    return mapping


def budget_from_path(path):
    m = re.search(r"results[_-]?(\d+)", path)
    if m:
        return int(m.group(1))
    base = os.path.basename(os.path.dirname(os.path.abspath(path)))
    ints = re.findall(r"(\d+)", base)
    return int(ints[-1]) if ints else None


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def read_norm(path, mapping, inv, have_budget_col, path_budget, drop_counter):
    """Yield cleaned chunks with logical column names and clean int seed/budget."""
    usecols = list(mapping.values())
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK,
                             low_memory=False):
        chunk = chunk.rename(columns=inv)
        chunk["seed"] = to_num(chunk["seed"])
        if have_budget_col:
            chunk["budget"] = to_num(chunk["budget"])
            if path_budget is not None:
                chunk["budget"] = chunk["budget"].fillna(path_budget)
        else:
            chunk["budget"] = path_budget
        for c in ("score", "stage1_score", "best_so_far"):
            if c in chunk.columns:
                chunk[c] = to_num(chunk[c])
        chunk["status"] = chunk["status"].astype(str).str.strip()
        before = len(chunk)
        chunk = chunk.dropna(subset=["seed", "budget"])
        drop_counter[0] += before - len(chunk)
        if len(chunk) == 0:
            continue
        chunk["seed"] = chunk["seed"].astype(int)
        chunk["budget"] = chunk["budget"].astype(int)
        yield chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="results/trace_summary")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    mapping = resolve_columns(args.inputs[0])
    inv = {v: k for k, v in mapping.items()}
    have_budget_col = "budget" in mapping
    has_stage1 = "stage1_score" in mapping
    has_score = "score" in mapping
    has_best = "best_so_far" in mapping

    # validate budget availability per file up front
    for p in args.inputs:
        if not have_budget_col and budget_from_path(p) is None:
            raise SystemExit(
                f"ERROR: no budget column and cannot parse a budget from the "
                f"path '{p}'. Put each file under a folder like "
                f"benchmark_results300/, or add a budget column.")

    status_by_cb = defaultdict(Counter)
    run_best, run_inc = {}, {}
    run_status_counts = defaultdict(Counter)
    run_n = Counter()
    pear = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])
    promo = defaultdict(lambda: [0, 0])
    reservoir = defaultdict(list)
    reservoir_seen = Counter()
    dropped = [0]

    # ---------- PASS A ----------
    print(">> Pass A: status vocabulary, incumbents, staged stats ...", flush=True)
    rows_seen = 0
    for path in args.inputs:
        pb = budget_from_path(path)
        for chunk in read_norm(path, mapping, inv, have_budget_col, pb, dropped):
            for r in chunk.itertuples(index=False):
                d = r._asdict()
                rk = (d["dataset"], d["config"], int(d["seed"]), int(d["budget"]))
                cb = (d["config"], int(d["budget"]))
                st = d["status"]
                status_by_cb[cb][st] += 1
                run_status_counts[rk][st] += 1
                run_n[rk] += 1

                full = d.get("score") if has_score else None
                s1 = d.get("stage1_score") if has_stage1 else None
                full_ok = full is not None and not (isinstance(full, float) and math.isnan(full))
                s1_ok = s1 is not None and not (isinstance(s1, float) and math.isnan(s1))

                key_val = d.get("best_so_far") if has_best else None
                if key_val is None or (isinstance(key_val, float) and math.isnan(key_val)):
                    key_val = full if full_ok else None
                if key_val is not None and not (isinstance(key_val, float) and math.isnan(key_val)):
                    if rk not in run_best or key_val > run_best[rk]:
                        run_best[rk] = key_val
                        run_inc[rk] = (st, full if full_ok else None,
                                       s1 if s1_ok else None)

                if s1_ok:
                    promo[cb][0] += 1
                    if full_ok:
                        promo[cb][1] += 1
                        p = pear[cb]
                        p[0] += 1; p[1] += s1; p[2] += full
                        p[3] += s1 * s1; p[4] += full * full; p[5] += s1 * full
                        reservoir_seen[cb] += 1
                        buf = reservoir[cb]
                        if len(buf) < RESERVOIR_CAP:
                            buf.append((round(s1, 6), round(full, 6)))
                        else:
                            j = random.randint(0, reservoir_seen[cb] - 1)
                            if j < RESERVOIR_CAP:
                                buf[j] = (round(s1, 6), round(full, 6))
                rows_seen += 1
            if rows_seen and rows_seen % 2_000_000 < CHUNK:
                print(f"   ... {rows_seen:,} rows", flush=True)
    print(f"   pass A done: {rows_seen:,} rows kept, {len(run_best):,} runs, "
          f"{dropped[0]:,} rows dropped for missing seed/budget", flush=True)

    # ---------- PASS B: false-negative rate ----------
    fn = defaultdict(lambda: [0, 0])
    if has_stage1 and has_score:
        print(">> Pass B: false-negative rate ...", flush=True)
        dropped_b = [0]
        for path in args.inputs:
            pb = budget_from_path(path)
            for chunk in read_norm(path, mapping, inv, have_budget_col, pb, dropped_b):
                sub = chunk[chunk["stage1_score"].notna() & chunk["score"].isna()]
                for r in sub.itertuples(index=False):
                    d = r._asdict()
                    rk = (d["dataset"], d["config"], int(d["seed"]), int(d["budget"]))
                    cb = (d["config"], int(d["budget"]))
                    fn[cb][0] += 1
                    inc = run_best.get(rk)
                    if inc is not None and d["stage1_score"] > inc:
                        fn[cb][1] += 1
        print("   pass B done", flush=True)

    # ---------- outputs ----------
    rows = []
    for (cfg, bud), c in sorted(status_by_cb.items()):
        for st, n in sorted(c.items()):
            rows.append(dict(config=cfg, budget=bud, status=st, count=n))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "status_summary.csv"), index=False)

    all_status = sorted({s for c in status_by_cb.values() for s in c})
    rrows = []
    for rk in sorted(run_best):
        dat, cfg, sd, bud = rk
        st, sc, s1 = run_inc.get(rk, (None, None, None))
        row = dict(dataset=dat, config=cfg, seed=sd, budget=bud,
                   incumbent_best=round(run_best[rk], 6), inc_status=st,
                   inc_score=(round(sc, 6) if sc is not None else ""),
                   inc_stage1=(round(s1, 6) if s1 is not None else ""),
                   inc_score_is_null=int(sc is None), n_rows=run_n[rk])
        for s in all_status:
            row[f"n_{s}"] = run_status_counts[rk].get(s, 0)
        rrows.append(row)
    pd.DataFrame(rrows).to_csv(os.path.join(args.out, "per_run_incumbent.csv"), index=False)

    mrows = []
    for cb in sorted(set(list(promo) + list(fn) + list(pear))):
        cfg, bud = cb
        n, sx, sy, sxx, syy, sxy = pear[cb]
        if n > 2:
            cov = sxy - sx * sy / n
            vx = sxx - sx * sx / n
            vy = syy - sy * sy / n
            pearson = cov / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float("nan")
        else:
            pearson = float("nan")
        # Spearman from the reservoir sample
        buf = reservoir.get(cb, [])
        if len(buf) >= 3:
            s1s, fulls = zip(*buf)
            sr, _ = spearmanr(s1s, fulls)
            spearman_v = round(sr, 4) if not (isinstance(sr, float) and math.isnan(sr)) else ""
        else:
            spearman_v = ""

        denom_s1, num_full = promo[cb]
        mrows.append(dict(
            config=cfg, budget=bud,
            pearson_stage1_full=(round(pearson, 4) if pearson == pearson else ""),
            spearman_stage1_full=spearman_v,
            n_pairs=n,
            promotion_rate=(round(num_full / denom_s1, 4) if denom_s1 else ""),
            promo_denom=denom_s1,
            false_neg_rate=(round(fn[cb][1] / fn[cb][0], 4) if fn[cb][0] else ""),
            not_promoted=fn[cb][0]))
    pd.DataFrame(mrows).to_csv(os.path.join(args.out, "staged_metrics.csv"), index=False)

    srows = []
    for (cfg, bud), buf in sorted(reservoir.items()):
        for s1, full in buf:
            srows.append(dict(config=cfg, budget=bud, stage1_score=s1, score=full))
    pd.DataFrame(srows).to_csv(os.path.join(args.out, "stage_full_sample.csv"), index=False)

    print("\n" + "=" * 70)
    print("STATUS VOCABULARY (overall):")
    overall = Counter()
    for c in status_by_cb.values():
        overall.update(c)
    for st, n in overall.most_common():
        print(f"   {st:>14}: {n:,}")
    print("\nSTAGED METRICS (config, budget): pearson | spearman | promo | false-neg")
    for m in mrows:
        print(f"   {m['config']:>10} {m['budget']:>4}s  r={m['pearson_stage1_full']!s:>7}"
              f"  rs={m['spearman_stage1_full']!s:>7}"
              f"  promo={m['promotion_rate']!s:>7}  fn={m['false_neg_rate']!s:>7}")
    if dropped[0]:
        print(f"\nNOTE: dropped {dropped[0]:,} rows with missing seed/budget.")
    print(f"\nWrote 4 files to ./{args.out}/  (zip and upload that folder)")
    print("=" * 70)


if __name__ == "__main__":
    main()
