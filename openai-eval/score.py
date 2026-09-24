"""Score runs produced by generate.py. Run this on Linux (e.g. WSL).

Uses the repo's own metrics, so numbers are comparable with the papers:
  - Decompile-Bench datasets: decompile-bench/metrics/cal_execute_rate.py, unchanged
  - Legacy HumanEval-Decompile: legacy_execute() below, a copy of evaluate_func in
    evaluation/run_evaluation_llm4decompile_vllm.py (that module needs vLLM just to import)
  - Edit similarity: decompile-bench/metrics/cal_edit_sim.py, unchanged

Usage:
  python3 openai-eval/score.py openai-eval/outputs/<run> [more runs...]
"""
import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "decompile-bench", "metrics"))

import bench_data  # noqa: E402
from cal_edit_sim import compute_ES  # noqa: E402
from cal_execute_rate import execute_rate  # noqa: E402
from tqdm import tqdm  # noqa: E402

OPTS = ["O0", "O1", "O2", "O3"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="Run directories written by generate.py")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--timeout", type=int, default=10, help="Seconds per compile and per test run")
    p.add_argument("--verbose", action="store_true", help="Show compiler errors and test output")
    return p.parse_args()


def silence_output():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)


def legacy_execute(c_include, c_func_decompile, c_test, timeout):
    """evaluate_func from evaluation/run_evaluation_llm4decompile_vllm.py, minus its include splitting
    (bench_data.normalize_legacy does that). "Compiled" means the function alone compiles with gcc -S."""
    flag_compile, flag_run = 0, 0
    c_combine = c_include + "\n" + c_func_decompile + "\n" + c_test
    c_onlyfunc = c_include + "\n" + c_func_decompile
    with tempfile.TemporaryDirectory() as temp_dir:
        pid = os.getpid()
        c_file = os.path.join(temp_dir, f"combine_{pid}.c")
        executable = os.path.join(temp_dir, f"combine_{pid}")
        c_file_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}.c")
        executable_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}")
        with open(c_file, "w") as f:
            f.write(c_combine)
        with open(c_file_onlyfunc, "w") as f:
            f.write(c_onlyfunc)
        try:
            subprocess.run(["gcc", "-S", c_file_onlyfunc, "-o", executable_onlyfunc, "-lm"], check=True, timeout=timeout)
            flag_compile = 1
        except Exception:  # noqa: BLE001
            return flag_compile, flag_run
        try:
            subprocess.run(["gcc", c_file, "-o", executable, "-lm"], check=True, timeout=timeout)
        except Exception:  # noqa: BLE001
            return flag_compile, flag_run
        try:
            subprocess.run([executable], capture_output=True, text=True, timeout=timeout, check=True)
            flag_run = 1
        except Exception:  # noqa: BLE001
            pass
    return flag_compile, flag_run


def score_one(task):
    idx, func_dep, code, test, language, timeout, legacy = task
    if legacy:
        comp, exe = legacy_execute(func_dep, code, test, timeout)
    else:
        # Same call as execute_rate_main in the repo: tests are always compiled at -O0.
        comp, exe = execute_rate(func_dep, code, test, timeout, language, "-O0")
    return idx, comp, exe


def load_generations(run_dir):
    gens, errors = {}, {}
    with open(os.path.join(run_dir, "generations.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("error"):
                gens[rec["idx"]] = rec  # later lines win
                errors.pop(rec["idx"], None)
            elif rec["idx"] not in gens:
                errors[rec["idx"]] = rec
    return gens, errors


def pct(num, den):
    return f"{100.0 * num / den:6.2f}" if den else "     -"


def score_run(run_dir, args):
    with open(os.path.join(run_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    data, info = bench_data.load(meta["dataset"])

    gens, errors = load_generations(run_dir)
    selected = meta.get("selected_indices") or list(range(len(data)))
    missing = [i for i in selected if i not in gens]
    tasks = [(i, data[i]["func_dep"], gens[i]["code"], data[i]["test"], data[i]["language"], args.timeout,
              info["legacy"])
             for i in selected if i in gens]

    print(f"\n=== {os.path.basename(os.path.normpath(run_dir))}")
    print(f"model: {meta['model']}  field: {meta['field']}  dataset: {meta['dataset']}")
    n_errors = sum(1 for i in missing if i in errors)
    if n_errors:
        print(f"Note: {n_errors} samples got an error from the server (often: ran out of --max-tokens); "
              f"they count as failures.")
    if len(missing) > n_errors:
        print(f"WARNING: {len(missing) - n_errors} of {len(selected)} selected samples have no generation yet; "
              f"they count as failures. Re-run generate.py to fill them in.")

    init = None if args.verbose else silence_output
    with multiprocessing.Pool(args.workers, initializer=init) as pool:
        results = list(tqdm(pool.imap_unordered(score_one, tasks), total=len(tasks), desc="compile+test"))

    per_sample = []
    for idx, comp, exe in sorted(results):
        es = compute_ES(data[idx]["func"], gens[idx]["code"]) if gens[idx]["code"].strip() else 0.0
        per_sample.append({"idx": idx, "opt": data[idx]["opt"], "language": data[idx]["language"],
                           "compiled": comp, "passed": exe, "edit_sim": es,
                           "finish_reason": gens[idx].get("finish_reason")})
    for idx in missing:
        per_sample.append({"idx": idx, "opt": data[idx]["opt"], "language": data[idx]["language"],
                           "compiled": 0, "passed": 0, "edit_sim": 0.0,
                           "finish_reason": "error" if idx in errors else "missing"})

    groups = defaultdict(list)
    for r in per_sample:
        groups[(r["language"], r["opt"])].append(r)
        groups[("all", r["opt"])].append(r)
        groups[(r["language"], "avg")].append(r)
        groups[("all", "avg")].append(r)

    summary = {}
    print(f"\n{'lang':<5}{'opt':<5}{'n':>6}{'compile%':>10}{'re-exec%':>10}{'edit sim':>10}")
    for lang in ["c", "cpp", "all"]:
        for opt in OPTS + ["avg"]:
            rows = groups.get((lang, opt))
            if not rows:
                continue
            n = len(rows)
            comp = sum(r["compiled"] for r in rows)
            exe = sum(r["passed"] for r in rows)
            es = sum(r["edit_sim"] for r in rows) / n
            summary[f"{lang}/{opt}"] = {"n": n, "compile_rate": comp / n, "reexec_rate": exe / n, "edit_sim": es}
            print(f"{lang:<5}{opt:<5}{n:>6}{pct(comp, n):>10}{pct(exe, n):>10}{es:>10.4f}")

    # The papers' AVG column is the unweighted mean of the O0-O3 rates.
    levels = [summary[f"all/{o}"] for o in OPTS if f"all/{o}" in summary]
    macro = {k: sum(s[k] for s in levels) / len(levels) for k in ("compile_rate", "reexec_rate", "edit_sim")}
    summary["all/paper_avg"] = macro
    print(f"\nPaper-style AVG (mean of O0-O3): re-exec {100 * macro['reexec_rate']:.2f}%, "
          f"compile {100 * macro['compile_rate']:.2f}%, edit sim {macro['edit_sim']:.4f}")

    truncated = sum(1 for r in per_sample if r["finish_reason"] == "length")
    if truncated:
        print(f"\nNote: {truncated} replies hit max_tokens (finish_reason=length); consider raising --max-tokens.")

    with open(os.path.join(run_dir, "scores.json"), "w", encoding="utf-8") as f:
        json.dump({"model": meta["model"], "field": meta["field"], "dataset": meta["dataset"],
                   "missing": len(missing), "summary": summary}, f, indent=2)
    with open(os.path.join(run_dir, "scores.jsonl"), "w", encoding="utf-8") as f:
        for r in sorted(per_sample, key=lambda r: r["idx"]):
            f.write(json.dumps(r) + "\n")
    return summary["all/avg"]


def main():
    args = parse_args()
    if os.name == "nt":
        sys.exit("Run score.py on Linux/WSL: the tests need Linux gcc/g++ headers and libcrypto.")
    overall = {run: score_run(run, args) for run in args.runs}
    if len(overall) > 1:
        print(f"\n{'run':<60}{'re-exec%':>10}{'edit sim':>10}")
        for run, s in overall.items():
            print(f"{os.path.basename(os.path.normpath(run)):<60}{100 * s['reexec_rate']:>10.2f}{s['edit_sim']:>10.4f}")


if __name__ == "__main__":
    main()
