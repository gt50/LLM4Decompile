# Evaluating local models through an OpenAI-compatible endpoint

The upstream scripts in `decompile-bench/` and `evaluation/` load models with vLLM on CUDA.
This directory swaps only the **generation** step for HTTP calls to any OpenAI-compatible
chat endpoint (llama.cpp `llama-server`, Unsloth Studio, LM Studio, Ollama, vLLM, ...).
**Scoring** reuses the repo's own `decompile-bench/metrics/` code unchanged, so results line
up with the paper's re-executability and edit-similarity numbers.

| File | Runs on | Purpose |
|---|---|---|
| `generate.py` | Windows or Linux, stdlib only | Sends each sample to the endpoint, saves replies (resumable) |
| `score.py` | Linux / WSL | Compiles each reply with the sample's tests; reports compile %, re-exec %, edit sim |
| `refine.py` | Windows (compiles via WSL) or Linux | Repairs a run's answers using compiler errors and side-by-side runs against the original (see step 4) |
| `execcheck.py` | Linux / WSL | Compile, link, run and disassemble helper used by `refine.py` |
| `bench_data.py` | | Loads both dataset formats (Decompile-Bench and legacy) |
| `prompts/default.txt` | | Prompt template (`{input}`, `{func_name}`, `{language}`, `{opt}`, `{input_kind}`, `{fence}`, `{deps}`). Asks for the same behavior, not the same machine code. |
| `prompts/exact.txt` | | Earlier prompt asking for code that "behaves exactly like the original"; thinking models tend to over-analyze optimized code with it. Use with `--prompt`. |

## 1. Point at your server

Unsloth Studio starts `llama-server` on a random port that changes whenever it loads a
model. To find the current port from PowerShell:

```powershell
(Get-CimInstance Win32_Process -Filter "Name='llama-server.exe'").CommandLine -replace '.*--port (\d+).*','$1'
```

Then use `--base-url http://127.0.0.1:<port>/v1`, or set `OPENAI_BASE_URL`. `--model`
defaults to the first model the server lists.

## 2. Generate

```powershell
# quick sanity check: 50 random samples, thinking off
py openai-eval/generate.py --base-url http://127.0.0.1:58434/v1 --sample 50 --no-think --max-tokens 4096 --tag nothink

# full HumanEval-Decompile (1312 samples = 164 problems x C/C++ x O0-O3) from Ghidra pseudo-code
py openai-eval/generate.py --base-url http://127.0.0.1:58434/v1 --field ghidra_pseudo
```

Output goes to `openai-eval/outputs/<dataset>-<field>-<model>[-<tag>]/`:
`generations.jsonl` (the full reply, reasoning, extracted code and token usage),
`meta.json` (all settings), and `<opt>/<idx>_<opt>.<c|cpp>` (the extracted code, in the same
layout the upstream `run_exe_rate.py` writes). Re-running the same command resumes a run
and retries any failed samples.

Useful options:

- `--field`: what the model sees. `asm` (objdump disassembly, which the paper's headline
  numbers use), `ghidra_pseudo`, `ida_pseudo`, `ghidra_asm`, `ida_asm`.
- `--dataset`: which benchmark to use.
  - `humaneval` (default) and `mbpp`: Decompile-Bench-Eval, with C and C++ (1312 and 7792 samples).
  - `legacy-asm` and `legacy-ghidra`: the original HumanEval-Decompile used in the
    LLM4Decompile V1.5/V2 paper tables (C only, 164 problems x O0-O3 = 656 samples).
    `legacy-asm` gives the model objdump disassembly, like the LLM4Decompile-End table.
    `legacy-ghidra` gives it Ghidra pseudo-code, like the "Ghidra + LLM4Decompile-Ref" table.
    Use these to compare against the paper's numbers.
- `--opt O0 O2`, `--language c`, `--sample N [--seed S]`, `--indices 0 5 9`: pick a subset.
- `--no-think`: disables reasoning for Qwen-style chat templates. Thinking models can spend
  thousands of tokens per sample; on an iGPU that is minutes per sample.
- `--workers`: concurrent requests, default 1. With llama.cpp, one request at a time is
  usually fastest per sample. Raise it only for servers built for batching, like vLLM.
- `--context-deps`: also shows the model the sample's `func_dep` (includes, helper
  functions). Useful to experiment with, but not comparable with the paper.
- `--extra-body '{"top_k": 20}'`: any extra request fields.

## 3. Score (in WSL)

One-time setup inside Ubuntu (WSL):

```bash
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev libboost-dev python3-venv
python3 -m venv ~/.venvs/llm4d-score
~/.venvs/llm4d-score/bin/pip install tqdm numpy editdistance
```

Then, from the repo root, which is under `/mnt/c/...` in WSL:

```bash
~/.venvs/llm4d-score/bin/python openai-eval/score.py openai-eval/outputs/<run> [<run2> ...]
```

This prints compile %, re-executability %, and edit similarity per language and
optimization level. It writes `scores.json` (the summary) and `scores.jsonl` (per-sample
results) into each run directory. Samples that are selected but have no generation count
as failures. Passing several runs adds a side-by-side comparison at the end.

## 4. Optional: repair with compiler and execution feedback

`refine.py` takes a finished `generate.py` run and tries to fix each answer, **without using the
benchmark's tests**:

1. **Compile check.** If the answer doesn't compile, the model gets the compiler errors, the
   original input and its previous code, and is asked for a corrected version.
2. **Side-by-side run.** The model writes a small test program (`prompts/driver.txt`) that calls
   the function on 8 inputs it chooses. That program is linked once against the *original*
   compiled function and once against the model's version, and both are run. The "original" is
   the dataset's reference source compiled at the sample's `-O` level, which is how the dataset's
   assembly was produced. It stands in for the binary you'd have in a real decompilation job;
   the model never sees its source.
3. **Signature check.** The argument registers each version reads on entry, and the functions
   it calls, are read from the disassembly and compared.

If the outputs differ, or the argument registers differ, the model gets the differing calls
(`prompts/repair_behavior.txt`) and tries again, up to `--rounds` times (default 3). The final
answer is the best round by these checks alone, and `score.py` scores it as usual.

```powershell
py openai-eval/refine.py openai-eval/outputs/<run> --base-url http://127.0.0.1:<port>/v1
```

Runs on Windows and compiles through WSL (`--wsl-distro`, default `Ubuntu-24.04`), or runs
directly on Linux. It writes `<run>-refine/`, containing `generations.jsonl` (final answers),
`rounds.jsonl` (every round's code, test program, outputs and checks) and `meta.json`.
Defaults are no-think with Unsloth's non-thinking sampling; `--think` turns reasoning on.
Report results as "with execution feedback": they are not single-attempt numbers like the
paper's tables.

## Notes on comparability

- The paper's `run_exe_rate.py` uses a bare completion prompt
  (`# This is the assembly code:\n...\n# What is the source code?\n`) because its models are
  fine-tuned for it. Chat models need instructions, so this harness uses
  `prompts/default.txt` and takes the code from the reply's fenced block. Differences in
  prompting are part of what you're measuring.
- Scoring is identical to upstream. For Decompile-Bench datasets, that's `func_dep + model output + test`,
  compiled at `-O0` with `gcc` or `g++ -std=c++17 ... -lm -lcrypto`, and run with a 10 s timeout.
  For the legacy datasets it mirrors `evaluate_func` in
  `evaluation/run_evaluation_llm4decompile_vllm.py`: `#include` lines are moved to the top, and
  "compile %" means the function alone compiles (`gcc -S`).
- The paper's AVG column is the mean of the O0-O3 rates. `score.py` prints it as
  "Paper-style AVG". The `all avg` row pools all samples instead; the two match only when each
  level has the same number of samples, which is true for full runs but not for `--sample`.
- Compiler versions differ from the authors', so expect small shifts in compile rates.
