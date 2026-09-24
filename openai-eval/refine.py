"""Repair decompilations with compiler and execution feedback, without using the benchmark's tests.

Starts from a generate.py run (round 0) and, for each sample, repeats up to --rounds times:

1. Compile the candidate. If it fails, send the compiler errors back and ask for a fix.
2. Ask the model for a small test driver that calls the function on inputs it chooses, then
   run that driver against the original compiled function and against the candidate
   (execcheck.py, through WSL on Windows). If the outputs differ, or the candidate's
   argument registers differ from the original's, send the differences back and ask for a fix.

The "original" is the dataset's reference source compiled at the sample's optimization level,
standing in for the binary you would have in a real decompilation job. Only its behavior and
disassembly are used; the model never sees the source, and the benchmark's tests are never run
or consulted. The final answer per sample is the best round by these checks alone.

Output: <out-dir>/<run>-refine[-<tag>]/ with generations.jsonl (final code, scored by score.py
as usual), rounds.jsonl (every round, driver and check result) and meta.json.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import bench_data
from generate import INPUT_KINDS, LANGUAGE_NAMES, build_prompt, extract_code, http_json

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="Run directory from generate.py to start from (its answers are round 0)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--model", default=None, help="Default: first model the server lists")
    p.add_argument("--rounds", type=int, default=3, help="Maximum repair rounds per sample")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--think", action="store_true", help="Leave reasoning on (default: send enable_thinking=false)")
    p.add_argument("--extra-body", default='{"top_k": 20, "presence_penalty": 1.5}',
                   help="JSON merged into every request")
    p.add_argument("--timeout", type=float, default=1800, help="Per-request timeout in seconds")
    p.add_argument("--indices", type=int, nargs="+", help="Only these indices (default: all in the run)")
    p.add_argument("--wsl-distro", default="Ubuntu-24.04", help="WSL distribution for compiling (Windows only)")
    p.add_argument("--out-dir", default=None, help="Default: next to the input run")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def fill(template, **values):
    """Substitute {name} placeholders in one pass, so inserted code is never re-scanned."""
    return PLACEHOLDER_RE.sub(lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0), template)


def read_prompt(name):
    with open(os.path.join(HERE, "prompts", name), "r", encoding="utf-8") as f:
        return f.read()


def to_wsl_path(path):
    path = os.path.abspath(path)
    drive, rest = os.path.splitdrive(path)
    return f"/mnt/{drive[0].lower()}{rest.replace(os.sep, '/')}"


def signature_of(code, func_name):
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", code, flags=re.S)
    m = re.search(rf"([^;{{}}#]*?\b{re.escape(func_name)}\s*\([^{{;]*\))\s*\{{", code)
    return " ".join(m.group(1).split()) if m else None


class Checker:
    def __init__(self, distro):
        script = os.path.join(HERE, "execcheck.py")
        self.cmd = (["wsl", "-d", distro, "--", "python3", to_wsl_path(script)] if os.name == "nt"
                    else [sys.executable, script])

    def __call__(self, sample, code, driver=None):
        req = {"func_dep": sample["func_dep"], "ref_func": sample["func"], "cand_code": code,
               "language": sample["language"], "opt": sample["opt"], "func_name": sample["func_name"],
               "driver": driver, "timeout": 10}
        p = subprocess.run(self.cmd, input=json.dumps(req), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=300)
        if p.returncode != 0 or not p.stdout.strip():
            raise RuntimeError(f"execcheck failed: {p.stderr[:500]}")
        return json.loads(p.stdout)


class Model:
    def __init__(self, args):
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        try:
            models = http_json(f"{self.base_url}/models", args.api_key, timeout=10).get("data", [])
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not reach {self.base_url}/models: {e}")
        self.name = args.model or (models[0]["id"] if models else None)
        if not self.name:
            sys.exit("Server lists no models; pass --model.")
        self.extra = json.loads(args.extra_body) if args.extra_body else {}

    def ask(self, prompt, system):
        body = {"model": self.name, "temperature": self.args.temperature, "top_p": self.args.top_p,
                "max_tokens": self.args.max_tokens, "stream": False,
                "messages": ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]}
        if not self.args.think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(self.extra)
        t0 = time.time()
        try:
            resp = http_json(f"{self.base_url}/chat/completions", self.args.api_key, body, timeout=self.args.timeout)
            choice = resp["choices"][0]
            content = choice.get("message", {}).get("content") or ""
            return {"content": content, "finish_reason": choice.get("finish_reason"),
                    "tokens": resp.get("usage", {}).get("completion_tokens", 0),
                    "elapsed_s": round(time.time() - t0, 1), "error": None}
        except Exception as e:  # noqa: BLE001
            return {"content": "", "finish_reason": None, "tokens": 0,
                    "elapsed_s": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}


def compare(res):
    """Return (consistent, n_diff, n_calls, arg_mismatch) for a check result with a driver run."""
    o, c = res["orig"], res["cand"]
    ol, cl = o["stdout"].splitlines(), c["stdout"].splitlines()
    n = max(len(ol), len(cl), 1)
    n_diff = sum(1 for k in range(n) if (ol[k] if k < len(ol) else None) != (cl[k] if k < len(cl) else None))
    oi, ci = res.get("orig_asm_info"), res.get("cand_asm_info")
    arg_mismatch = bool(oi and ci and (oi["int_arg_regs"] != ci["int_arg_regs"]
                                       or oi["float_arg_regs"] != ci["float_arg_regs"]))
    consistent = n_diff == 0 and o["status"] == c["status"] and not arg_mismatch
    return consistent, n_diff, n, arg_mismatch


def describe_args(info):
    ints, floats = info["int_arg_regs"], info["float_arg_regs"]
    parts = [f"{len(ints)} integer/pointer argument(s)" + (f" ({', '.join(ints)})" if ints else ""),
             f"{len(floats)} floating-point argument(s)" + (f" ({', '.join(floats)})" if floats else "")]
    return " and ".join(parts)


def behavior_feedback(res):
    o, c = res["orig"], res["cand"]
    ol, cl = o["stdout"].splitlines(), c["stdout"].splitlines()
    lines = []
    diffs = []
    for k in range(max(len(ol), len(cl))):
        a = ol[k] if k < len(ol) else "(no output)"
        b = cl[k] if k < len(cl) else "(no output)"
        if a != b:
            diffs.append((a, b))
    if diffs:
        lines.append(f"{len(diffs)} of {max(len(ol), len(cl))} calls give different results. "
                     f"Each pair is the same call, first to the original, then to the attempt:")
        for a, b in diffs[:6]:
            lines.append(f"  original: {a}\n  attempt:  {b}")
    if o["status"] != c["status"]:
        lines.append(f"The program calling the original finished with: {o['status']}. "
                     f"The program calling the attempt finished with: {c['status']}.")
    oi, ci = res.get("orig_asm_info"), res.get("cand_asm_info")
    if oi and ci:
        if oi["int_arg_regs"] != ci["int_arg_regs"] or oi["float_arg_regs"] != ci["float_arg_regs"]:
            lines.append(f"Signature check: the original function reads {describe_args(oi)} on entry; "
                         f"the attempt reads {describe_args(ci)}. The parameter list is probably wrong, "
                         f"which can also make the outputs above misleading.")
        missing = [x for x in oi["calls"] if x not in ci["calls"]]
        extra = [x for x in ci["calls"] if x not in oi["calls"]]
        if missing:
            lines.append(f"The original calls {', '.join(missing)}; the attempt does not.")
        if extra:
            lines.append(f"The attempt calls {', '.join(extra)}; the original does not.")
    return "\n".join(lines)


def main():
    args = parse_args()
    with open(os.path.join(args.run, "meta.json"), "r", encoding="utf-8") as f:
        base_meta = json.load(f)
    data, info = bench_data.load(base_meta["dataset"])
    field = base_meta["field"]
    system = base_meta.get("system", "")
    base_template = base_meta.get("prompt_template") or read_prompt("default.txt")
    driver_tpl, compile_tpl, behavior_tpl = (read_prompt(n) for n in
                                             ("driver.txt", "repair_compile.txt", "repair_behavior.txt"))

    round0 = {}
    with open(os.path.join(args.run, "generations.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if not rec.get("error"):
                round0[rec["idx"]] = rec
    selected = args.indices or base_meta.get("selected_indices") or sorted(round0)

    model = Model(args)
    check = Checker(args.wsl_distro)
    run_name = os.path.basename(os.path.normpath(args.run)) + "-refine" + (f"-{args.tag}" if args.tag else "")
    out_dir = os.path.join(args.out_dir or os.path.dirname(os.path.abspath(args.run)), run_name)
    os.makedirs(out_dir, exist_ok=True)
    gen_path, rounds_path = os.path.join(out_dir, "generations.jsonl"), os.path.join(out_dir, "rounds.jsonl")

    meta = dict(base_meta)
    meta.update({"model": model.name, "baseline_run": os.path.abspath(args.run), "baseline_model": base_meta["model"],
                 "selected_indices": selected,
                 "refine": {"rounds": args.rounds, "temperature": args.temperature, "top_p": args.top_p,
                            "max_tokens": args.max_tokens, "think": args.think, "extra_body": model.extra,
                            "driver_prompt": driver_tpl, "repair_compile_prompt": compile_tpl,
                            "repair_behavior_prompt": behavior_tpl}})
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    done = set()
    if os.path.exists(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f:
            done = {json.loads(line)["idx"] for line in f}
    pending = [i for i in selected if i not in done]
    print(f"model:   {model.name}\nstart:   {args.run}\nrun dir: {out_dir}\n"
          f"selected {len(selected)}, already done {len(done & set(selected))}, to refine {len(pending)}\n", flush=True)

    t_start = time.time()
    for n, idx in enumerate(pending, 1):
        sample = data[idx]
        lang = LANGUAGE_NAMES.get(sample["language"], sample["language"])
        common = {"input_kind": INPUT_KINDS[field], "func_name": sample["func_name"], "language": lang,
                  "opt": sample["opt"], "fence": sample["language"], "input": sample[field].strip()}
        history = []
        tokens = 0

        if idx in round0:
            code = round0[idx]["code"]
            finish = round0[idx].get("finish_reason")
        else:  # no usable round-0 answer: generate one with the run's own prompt
            r = model.ask(build_prompt(base_template, sample, field, base_meta.get("context_deps")), system)
            code, finish, tokens = extract_code(r["content"], sample["func_name"]), r["finish_reason"], r["tokens"]

        driver, driver_sig, driver_note = None, None, None
        for rnd in range(args.rounds + 1):
            entry = {"round": rnd, "code": code, "finish_reason": finish}
            history.append(entry)
            res = check(sample, code)
            entry["compile_ok"] = res["compile_ok"]
            if res.get("internal_error"):
                entry["note"] = res["internal_error"]
                break
            if not res["compile_ok"]:
                entry["compile_errors"] = res["compile_errors"]
                if rnd == args.rounds:
                    break
                errors = res["compile_errors"] or "(no compiler output)"
                if sample["func_name"] not in code:
                    errors = f"The attempt does not contain a definition of {sample['func_name']}.\n" + errors
                r = model.ask(fill(compile_tpl, code=code, errors=errors, **common), system)
                entry["next"] = "compile"
            else:
                sig = signature_of(code, sample["func_name"])
                if driver is None or sig != driver_sig:
                    driver, driver_sig, driver_note = None, sig, None
                    driver_prompt = fill(driver_tpl, code=code, signature=sig or sample["func_name"] + "(...)",
                                         **common)
                    for _ in range(2):
                        d = model.ask(driver_prompt, "")
                        tokens += d["tokens"]
                        candidate_driver = extract_code(d["content"], "main")
                        dres = check(sample, code, candidate_driver)
                        if dres.get("driver_ok"):
                            driver, res = candidate_driver, dres
                            break
                        driver_note = dres.get("driver_errors") or d.get("error") or "driver failed"
                        driver_prompt = fill(driver_tpl, code=code, signature=sig or sample["func_name"] + "(...)",
                                             **common) + (f"\n\nYour previous test program failed to build:\n```\n"
                                                          f"{driver_note}\n```\nFix it.")
                    if driver is None:
                        entry["note"] = "could not build a test driver: " + driver_note[:300]
                        break
                else:
                    res = check(sample, code, driver)
                entry["driver"] = driver
                entry["orig"], entry["cand"] = res["orig"], res["cand"]
                entry["orig_asm_info"], entry["cand_asm_info"] = res["orig_asm_info"], res["cand_asm_info"]
                consistent, n_diff, n_calls, arg_mismatch = compare(res)
                entry.update({"consistent": consistent, "n_diff": n_diff, "n_calls": n_calls,
                              "arg_mismatch": arg_mismatch})
                if consistent or rnd == args.rounds:
                    break
                r = model.ask(fill(behavior_tpl, code=code, feedback=behavior_feedback(res), **common), system)
                entry["next"] = "behavior"
            tokens += r["tokens"]
            if r["error"]:
                entry["note"] = "repair request failed: " + r["error"]
                break
            code, finish = extract_code(r["content"], sample["func_name"]), r["finish_reason"]

        def rank(e):
            frac = e["n_diff"] / e["n_calls"] if "n_diff" in e else 1.0
            return (bool(e.get("consistent")), bool(e.get("compile_ok")), -frac, e["round"])

        best = max(history, key=rank)
        final = {"idx": idx, "opt": sample["opt"], "language": sample["language"], "func_name": sample["func_name"],
                 "code": best["code"], "finish_reason": best.get("finish_reason"), "error": None,
                 "chosen_round": best["round"], "rounds_run": len(history) - 1,
                 "consistent": bool(best.get("consistent")), "tokens": tokens}
        with open(gen_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(final, ensure_ascii=False) + "\n")
        with open(rounds_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"idx": idx, "history": history}, ensure_ascii=False) + "\n")

        def short(e):
            if not e.get("compile_ok"):
                return "no-compile"
            if "n_diff" not in e:
                return e.get("note", "?")[:30]
            return "match" if e["consistent"] else (f"{e['n_diff']}/{e['n_calls']} differ"
                                                    + (" +args" if e["arg_mismatch"] else ""))
        trail = " -> ".join(short(e) for e in history)
        eta = (time.time() - t_start) / n * (len(pending) - n) / 60
        print(f"[{n}/{len(pending)}] idx={idx} {sample['opt']}: {trail}  => round {best['round']}"
              f"  ({tokens} tok, eta {eta:.0f} min)", flush=True)

    print(f"\nDone in {(time.time() - t_start) / 60:.1f} min. Score in WSL:  python3 openai-eval/score.py "
          f"{os.path.relpath(out_dir, os.path.dirname(HERE)).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
