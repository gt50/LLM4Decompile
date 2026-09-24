"""Generate decompilations from any OpenAI-compatible chat endpoint.

Works with llama.cpp's llama-server (including the one Unsloth Studio launches),
LM Studio, Ollama, vLLM, etc. Uses only the Python standard library, so it runs
from any Python >= 3.8 on Windows or Linux.

Results are appended to <out-dir>/<run-name>/generations.jsonl, one line per
sample, so an interrupted run can be resumed by re-running the same command.
Score the run afterwards with score.py (inside WSL/Linux).
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import bench_data

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

INPUT_KINDS = {
    "asm": "x86-64 disassembly (objdump, AT&T syntax)",
    "ghidra_pseudo": "Ghidra decompiler pseudo-C output",
    "ida_pseudo": "IDA Pro (Hex-Rays) decompiler pseudo-C output",
    "ghidra_asm": "x86-64 disassembly from Ghidra",
    "ida_asm": "x86-64 disassembly from IDA Pro",
}

LANGUAGE_NAMES = {"c": "C", "cpp": "C++"}

FENCE_RE = re.compile(r"```[^\n`]*\n(.*?)```", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"),
                   help="OpenAI-compatible base URL, e.g. http://127.0.0.1:58434/v1 (env: OPENAI_BASE_URL)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="Bearer token, if the server needs one (env: OPENAI_API_KEY)")
    p.add_argument("--model", default=None, help="Model id to request (default: first id from /models)")
    p.add_argument("--dataset", default="humaneval",
                   help="'humaneval', 'mbpp' (Decompile-Bench), 'legacy-asm', 'legacy-ghidra' (the V1.5/V2 paper's "
                        "HumanEval-Decompile), or a path to a JSON file in either format")
    p.add_argument("--field", default=None, choices=sorted(INPUT_KINDS),
                   help="Which input the model sees (default: asm; legacy datasets have only one)")
    p.add_argument("--prompt", default=os.path.join(HERE, "prompts", "default.txt"), help="Prompt template file")
    p.add_argument("--system", default="You are an expert reverse engineer who turns compiled code back into clean, correct source code.",
                   help="System message ('' to omit)")
    p.add_argument("--context-deps", action="store_true",
                   help="Also show the model the sample's func_dep (includes/helpers). Not comparable with the paper's numbers.")
    p.add_argument("--opt", nargs="+", choices=["O0", "O1", "O2", "O3"], help="Only these optimization levels")
    p.add_argument("--language", nargs="+", choices=["c", "cpp"], help="Only these languages")
    p.add_argument("--sample", type=int, default=None, help="Random subset of N samples (after filters)")
    p.add_argument("--seed", type=int, default=0, help="Seed for --sample")
    p.add_argument("--indices", type=int, nargs="+", help="Only these dataset indices")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=16384, help="Includes reasoning tokens for thinking models")
    p.add_argument("--no-think", action="store_true",
                   help="Send chat_template_kwargs.enable_thinking=false (Qwen-style templates)")
    p.add_argument("--extra-body", default=None, help="JSON object merged into every request body")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent requests. llama.cpp is usually fastest one at a time; only raise this for "
                        "servers built for batching (vLLM, etc.)")
    p.add_argument("--timeout", type=float, default=1800, help="Per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=3, help="Retries for network errors/timeouts")
    p.add_argument("--retry-errors", action="store_true",
                   help="On resume, also redo samples the server answered with an error (skipped by default)")
    p.add_argument("--out-dir", default=os.path.join(HERE, "outputs"))
    p.add_argument("--run-name", default=None, help="Default: <dataset>-<field>-<model>[-<tag>]")
    p.add_argument("--tag", default=None, help="Suffix for the default run name, e.g. 'nothink'")
    return p.parse_args()


def http_json(url, api_key, body=None, timeout=60):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def build_prompt(template, sample, field, context_deps):
    lang = LANGUAGE_NAMES.get(sample["language"], sample["language"])
    deps = ""
    if context_deps and sample["func_dep"].strip():
        deps = ("The following code is already present in the file before `{name}` "
                "(do not repeat it):\n\n```{fence}\n{dep}\n```\n\n").format(
            name=sample["func_name"], fence=sample["language"], dep=sample["func_dep"].strip())
    text = (template
            .replace("{input_kind}", INPUT_KINDS[field])
            .replace("{func_name}", sample["func_name"])
            .replace("{language}", lang)
            .replace("{fence}", sample["language"])
            .replace("{opt}", sample["opt"])
            .replace("{deps}", deps))
    # Substitute the input last so its contents are never re-scanned for placeholders.
    return text.replace("{input}", sample[field].strip())


def extract_code(content, func_name):
    """Pull the C/C++ code out of a chat reply."""
    content = THINK_RE.sub("", content).strip()
    blocks = [b.strip() for b in FENCE_RE.findall(content)]
    if blocks:
        with_name = [b for b in blocks if func_name in b]
        return (with_name or [max(blocks, key=len)])[-1]
    # Unterminated fence (e.g. reply cut off at max_tokens): take everything after it.
    if "```" in content:
        return content.split("```", 1)[1].split("\n", 1)[-1].strip()
    return content


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    try:
        models = http_json(f"{base_url}/models", args.api_key, timeout=10).get("data", [])
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Could not reach {base_url}/models: {e}")
    model = args.model or (models[0]["id"] if models else None)
    if not model:
        sys.exit("Server lists no models; pass --model.")
    model_meta = next((m for m in models if m.get("id") == model), {})

    data, info = bench_data.load(args.dataset)
    dataset_stored, dataset_name = info["stored"], info["name"]
    if args.field is None:
        args.field = info["fields"][0] if info["legacy"] else "asm"
    if args.field not in info["fields"]:
        sys.exit(f"--field {args.field} is not available in {dataset_stored}; it has: {', '.join(info['fields'])}")

    todo = list(range(len(data)))
    if args.indices:
        todo = [i for i in args.indices if 0 <= i < len(data)]
    if args.opt:
        todo = [i for i in todo if data[i]["opt"] in args.opt]
    if args.language:
        todo = [i for i in todo if data[i]["language"] in args.language]
    if args.sample and args.sample < len(todo):
        todo = sorted(random.Random(args.seed).sample(todo, args.sample))

    run_name = args.run_name or "-".join(
        filter(None, [dataset_name, None if info["legacy"] else args.field, slug(model.split("/")[-1]), args.tag]))
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    gen_path = os.path.join(run_dir, "generations.jsonl")

    done = set()
    if os.path.exists(gen_path):
        with open(gen_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("error") or (rec.get("error_kind") == "server" and not args.retry_errors):
                    done.add(rec["idx"])
    pending = [i for i in todo if i not in done]

    with open(args.prompt, "r", encoding="utf-8") as f:
        template = f.read()

    extra = json.loads(args.extra_body) if args.extra_body else {}
    meta = {
        "model": model,
        "model_meta": model_meta,
        "base_url": base_url,
        "dataset": dataset_stored,
        "legacy": info["legacy"],
        "field": args.field,
        "prompt_template": template,
        "system": args.system,
        "context_deps": args.context_deps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "no_think": args.no_think,
        "extra_body": extra,
        "selected_indices": todo,
    }
    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"model:   {model}\nserver:  {base_url}\ndataset: {dataset_stored} ({len(data)} samples)\n"
          f"field:   {args.field}\nrun dir: {run_dir}\n"
          f"selected {len(todo)}, already done {len(todo) - len(pending)}, to generate {len(pending)}\n")
    if not pending:
        return

    lock = threading.Lock()
    counter = {"n": 0, "ok": 0}
    t_start = time.time()

    def run_one(idx):
        sample = data[idx]
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": build_prompt(template, sample, args.field, args.context_deps)})
        body = {"model": model, "messages": messages, "temperature": args.temperature,
                "max_tokens": args.max_tokens, "stream": False}
        if args.top_p is not None:
            body["top_p"] = args.top_p
        if args.no_think:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        body.update(extra)

        rec = {"idx": idx, "opt": sample["opt"], "language": sample["language"], "func_name": sample["func_name"]}
        t0 = time.time()
        for attempt in range(args.retries + 1):
            try:
                resp = http_json(f"{base_url}/chat/completions", args.api_key, body, timeout=args.timeout)
                choice = resp["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                rec.update({
                    "code": extract_code(content, sample["func_name"]),
                    "raw": content,
                    "reasoning": msg.get("reasoning_content") or "",
                    "finish_reason": choice.get("finish_reason"),
                    "usage": resp.get("usage", {}),
                    "error": None,
                })
                break
            except (OSError, KeyError, IndexError, ValueError) as e:  # URLError/timeouts are OSError
                server_error = isinstance(e, urllib.error.HTTPError)
                detail = e.read().decode("utf-8", "replace")[:500] if server_error else ""
                rec["error"] = f"{type(e).__name__}: {e} {detail}".strip()
                # The server answered with an error (e.g. llama.cpp failing at the token limit):
                # retrying the same request usually just burns the same time again.
                rec["error_kind"] = "server" if server_error else "network"
                if server_error:
                    break
                if attempt < args.retries:
                    time.sleep(2 ** attempt)
        rec["elapsed_s"] = round(time.time() - t0, 2)

        with lock:
            with open(gen_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if not rec["error"]:
                opt_dir = os.path.join(run_dir, sample["opt"])
                os.makedirs(opt_dir, exist_ok=True)
                with open(os.path.join(opt_dir, f"{idx}_{sample['opt']}.{sample['language']}"), "w", encoding="utf-8") as f:
                    f.write(rec["code"])
                counter["ok"] += 1
            counter["n"] += 1
            n = counter["n"]
            elapsed = time.time() - t_start
            eta = elapsed / n * (len(pending) - n)
            toks = rec.get("usage", {}).get("completion_tokens", "?")
            status = rec["error"] or (rec["finish_reason"] if rec["finish_reason"] != "stop" else "ok")
            print(f"[{n}/{len(pending)}] idx={idx} {sample['opt']} {sample['language']:<3} "
                  f"{rec['elapsed_s']:>6.1f}s {toks:>6} tok  {status}  (eta {eta / 60:.1f} min)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, i) for i in pending]
        try:
            for fut in as_completed(futures):
                fut.result()
        except KeyboardInterrupt:
            print("\nInterrupted; finished samples are saved. Re-run the same command to resume.")
            pool.shutdown(wait=False, cancel_futures=True)
            sys.exit(130)

    print(f"\nDone: {counter['ok']}/{len(pending)} succeeded in {(time.time() - t_start) / 60:.1f} min.")
    print(f"Score in WSL:  python3 openai-eval/score.py {os.path.relpath(run_dir, REPO_ROOT).replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
