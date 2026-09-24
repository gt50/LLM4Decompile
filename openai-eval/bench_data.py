"""Dataset loading shared by generate.py and score.py (standard library only).

Two formats are supported and normalized to the Decompile-Bench record layout
(index, func_name, func_dep, func, test, opt, language, <input fields>):

- Decompile-Bench-Eval (decompile-bench/data/*.json): used as-is.
- Legacy HumanEval-Decompile (legacy-test/*.json, used for the LLM4Decompile V1.5/V2
  paper tables): fields task_id/type/c_func/c_test/input_asm_prompt. The #include
  lines are split out exactly the way evaluation/run_evaluation_llm4decompile_vllm.py
  does, so func_dep + "\\n" + output + "\\n" + test reproduces its c_combine.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# name -> (path relative to the repo, input field for legacy files)
DATASETS = {
    "humaneval": ("decompile-bench/data/humaneval-decompile.json", None),
    "mbpp": ("decompile-bench/data/mbpp-decompile.json", None),
    "legacy-asm": ("legacy-test/decompile-eval-executable-gcc-obj.json", "asm"),
    "legacy-ghidra": ("legacy-test/decompile-eval-executable-gcc-ghidra.json", "ghidra_pseudo"),
}


def resolve(name):
    """Return (absolute path, path to store in meta.json, short name, legacy input field or None)."""
    if name in DATASETS:
        rel, legacy_field = DATASETS[name]
        path, short = os.path.join(REPO_ROOT, rel), name
    else:
        path = name if os.path.isabs(name) else os.path.join(REPO_ROOT, name)
        if not os.path.exists(path):
            path = os.path.abspath(name)
        short, legacy_field = None, None
        for key, (rel, field) in DATASETS.items():
            if os.path.normcase(os.path.abspath(path)) == os.path.normcase(os.path.join(REPO_ROOT, rel)):
                short, legacy_field = key, field
        if short is None:
            short = os.path.basename(path).replace("-decompile.json", "").replace(".json", "")
            legacy_field = "ghidra_pseudo" if "ghidra" in short else "asm"  # only used if the file is legacy
    # Stored repo-relative (posix) when possible so score.py can find it from WSL.
    try:
        rel = os.path.relpath(path, REPO_ROOT)
        stored = path if rel.startswith("..") else rel.replace(os.sep, "/")
    except ValueError:  # different drive on Windows
        stored = path
    return path, stored, short, legacy_field


def normalize_legacy(rec, index, field):
    c_func, c_test = rec["c_func"], rec["c_test"]
    c_include = ""
    for line in c_func.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_func = c_func.replace(line, "")
    for line in c_test.split("\n"):
        if "#include" in line:
            c_include += line + "\n"
            c_test = c_test.replace(line, "")
    return {
        "index": index,
        "func_name": "func0",
        "func_dep": c_include,
        "func": c_func.strip(),
        "test": c_test,
        "opt": rec["type"],
        "language": "c",
        field: rec["input_asm_prompt"],
    }


def load(name):
    """Return (records, info) where info has path, stored, name, legacy, fields."""
    path, stored, short, legacy_field = resolve(name)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    legacy = bool(raw) and "c_func" in raw[0]
    if legacy:
        data = [normalize_legacy(r, i, legacy_field) for i, r in enumerate(raw)]
        fields = [legacy_field]
    else:
        data = raw
        fields = [k for k in ("asm", "ghidra_pseudo", "ida_pseudo", "ghidra_asm", "ida_asm") if raw and k in raw[0]]
    return data, {"path": path, "stored": stored, "name": short, "legacy": legacy, "fields": fields}
