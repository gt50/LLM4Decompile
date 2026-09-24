"""Linux-side checks for refine.py: compile a candidate, and run it side by side with the original.

Reads one JSON request on stdin and prints one JSON result. Called through WSL from Windows
(or directly on Linux). Standard library only; needs gcc/g++ and objdump.

The "original binary" is rebuilt from the dataset's reference source at the sample's
optimization level, which is how the dataset's assembly was produced. It is only ever
linked and executed; its source is never shown to the model. The benchmark's tests are
not used here at all.

Request:  {"func_dep", "ref_func", "cand_code", "language", "opt", "driver" (optional), "timeout"}
Result:   {"compile_ok", "compile_errors", "driver_ok", "driver_errors",
           "orig": {"stdout", "status"}, "cand": {...}, "orig_asm_info", "cand_asm_info"}
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ARG_REGS_INT = [("rdi", "edi", "di", "dil"), ("rsi", "esi", "si", "sil"), ("rdx", "edx", "dx", "dl"),
                ("rcx", "ecx", "cx", "cl"), ("r8", "r8d", "r8w", "r8b"), ("r9", "r9d", "r9w", "r9b")]
ARG_REGS_FLOAT = [f"xmm{i}" for i in range(8)]
# Instructions whose last operand is only written, not read.
WRITE_ONLY = re.compile(r"^(mov|lea|cvt|pxor|xorps|xorpd|set|pop|movs|movz|movabs|cmov)")
MAX_OUT = 6000


def run(cmd, timeout, cwd=None):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, errors="replace")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return "timeout", "", ""


def compiler(language):
    if language == "cpp":
        return ["g++", "-std=c++17", "-w"], ["-lm", "-lcrypto"]
    return ["gcc", "-w"], ["-lm"]


def clean_errors(stderr, workdir):
    lines = []
    for line in stderr.replace(workdir + "/", "").splitlines():
        if "error" in line or "undefined reference" in line or line.startswith(" "):
            lines.append(line)
    return "\n".join(lines[:25])


def func_asm(binary, func_name, timeout):
    """objdump the binary and return the cleaned instructions of func_name (dataset format)."""
    code, out, _ = run(["objdump", "-d", "-C", "--no-show-raw-insn", binary], timeout)
    if code != 0:
        return ""
    # C symbols appear as <func0>, demangled C++ ones as <func0(std::vector<int, ...>)>.
    m = re.search(rf"^[0-9a-f]+ <{re.escape(func_name)}(?:\(.*\))?>:\n(.*?)(?:\n\n|\Z)", out, re.S | re.M)
    if not m:
        return ""
    insns = []
    for line in m.group(1).splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2:
            insns.append(parts[1].split("#")[0].strip())
    return "\n".join(insns)


def asm_info(asm):
    """Rough signature and call summary from x86-64 AT&T assembly."""
    calls = sorted({m.group(1).split("@")[0] for m in re.finditer(r"call\w*\s+\S+\s+<([^>+(]+)", asm)}
                   - {"__stack_chk_fail"})
    int_args, float_args = [], []
    reg_first_use = {}
    for insn in asm.splitlines():
        mnemonic, _, ops = insn.partition(" ")
        ops = ops.strip()
        operands = [o.strip() for o in re.split(r",(?![^(]*\))", ops)] if ops else []
        for idx, names in enumerate(ARG_REGS_INT):
            if idx in reg_first_use:
                continue
            for op_i, op in enumerate(operands):
                if any(re.search(rf"%{n}\b", op) for n in names):
                    is_dest = op_i == len(operands) - 1 and op.startswith("%")
                    write_only = is_dest and WRITE_ONLY.match(mnemonic) and len(operands) > 1
                    zeroing = mnemonic.startswith("xor") and len(set(operands)) == 1
                    reg_first_use[idx] = not (write_only or zeroing)
                    break
        for idx, name in enumerate(ARG_REGS_FLOAT):
            key = f"x{idx}"
            if key in reg_first_use:
                continue
            for op_i, op in enumerate(operands):
                if re.search(rf"%{name}\b", op):
                    is_dest = op_i == len(operands) - 1
                    write_only = is_dest and WRITE_ONLY.match(mnemonic) and len(operands) > 1
                    zeroing = len(operands) == 2 and operands[0] == operands[1]
                    reg_first_use[key] = not (write_only or zeroing)
                    break
        if mnemonic.startswith("call"):
            break  # after a call, argument registers hold the callee's values
    int_args = [ARG_REGS_INT[i][0] for i in range(6) if reg_first_use.get(i)]
    float_args = [ARG_REGS_FLOAT[i] for i in range(8) if reg_first_use.get(f"x{i}")]
    return {"calls": calls, "int_arg_regs": int_args, "float_arg_regs": float_args}


def main():
    req = json.load(sys.stdin)
    t = req.get("timeout", 10)
    lang = req["language"]
    ext = "cpp" if lang == "cpp" else "c"
    cc, libs = compiler(lang)
    opt = "-" + req["opt"]
    res = {"compile_ok": False, "compile_errors": "", "driver_ok": None, "driver_errors": "",
           "orig": None, "cand": None, "orig_asm_info": None, "cand_asm_info": None}

    with tempfile.TemporaryDirectory() as d:
        def write(name, text):
            with open(os.path.join(d, name), "w") as f:
                f.write(text)

        write(f"orig.{ext}", req["func_dep"] + "\n" + req["ref_func"] + "\n")
        write(f"cand.{ext}", req["func_dep"] + "\n" + req["cand_code"] + "\n")
        code, _, err = run(cc + [opt, "-c", f"orig.{ext}", "-o", "orig.o"], t, d)
        if code != 0:
            res["internal_error"] = "reference source failed to compile: " + err[:500]
            print(json.dumps(res))
            return
        code, _, err = run(cc + [opt, "-c", f"cand.{ext}", "-o", "cand.o"], t, d)
        res["compile_ok"] = code == 0
        if not res["compile_ok"]:
            res["compile_errors"] = clean_errors(err, d) if code != "timeout" else "compiler timed out"
            print(json.dumps(res))
            return
        if not req.get("driver"):
            print(json.dumps(res))
            return

        write(f"driver.{ext}", req["driver"])
        code, _, err = run(cc + ["-O0", f"driver.{ext}", "orig.o", "-o", "d_orig"] + libs, t, d)
        if code != 0:
            res["driver_ok"] = False
            res["driver_errors"] = clean_errors(err, d) if code != "timeout" else "compiler timed out"
            print(json.dumps(res))
            return
        code, _, err = run(cc + ["-O0", f"driver.{ext}", "cand.o", "-o", "d_cand"] + libs, t, d)
        if code != 0:
            # The driver builds against the original but not against the candidate: usually the
            # candidate is missing something the driver needs, e.g. a different signature in C++.
            res["driver_ok"] = True
            res["cand"] = {"stdout": "", "status": "link failed: " + clean_errors(err, d)}
        res["driver_ok"] = True
        for name, binary in (("orig", "d_orig"), ("cand", "d_cand")):
            if res.get(name):
                continue
            # Unbuffered stdout so lines printed before a crash are not lost.
            code, out, _ = run(["stdbuf", "-o0", "./" + binary], t, d)
            if code == "timeout":
                status = "timed out"
            elif code < 0:
                status = f"crashed (signal {-code})"
            elif code != 0:
                status = f"exit code {code}"
            else:
                status = "ok"
            res[name] = {"stdout": out[:MAX_OUT], "status": status}
        res["orig_asm_info"] = asm_info(func_asm(os.path.join(d, "d_orig"), req.get("func_name", "func0"), t))
        if os.path.exists(os.path.join(d, "d_cand")):
            res["cand_asm_info"] = asm_info(func_asm(os.path.join(d, "d_cand"), req.get("func_name", "func0"), t))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
