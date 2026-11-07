"""Generate *_x.py variants that run extra CV repeats: env R0 (first repeat index), NREP (number of repeats),
SUF (suffix of the results directories that xclf/ridge read their saved subsets from)."""
import re
HDR = "import os as _os\nR0 = int(_os.environ.get('R0', '0')); NREP = int(_os.environ.get('NREP', '2')); SUF = _os.environ.get('SUF', '')\n"
for f in ["run_wrapper", "run_cap", "run_emt", "run_xclf", "run_ridge", "run_thr", "run_tomek"]:
    s = open(f + ".py").read()
    m = list(re.finditer(r"^(import|from) .*\n", s, re.M))[-1]
    s = s[:m.end()] + HDR + s[m.end():]
    s = s.replace("n_repeats=2", "n_repeats=NREP").replace("repeats=2", "repeats=NREP")
    s = s.replace("seed + r)", "seed + r + R0)").replace("[seed, r,", "[seed, r + R0,")
    s = re.sub(r'\("(results_\w+)"', r'("\1" + SUF', s)
    s = re.sub(r'f"(results_\w+)/\{name\}', r'f"\1{SUF}/{name}', s)
    s = s.replace("from run_thr import", "from run_thr_x import").replace("from run_wrapper import", "from run_wrapper_x import")
    open(f + "_x.py", "w").write(s)
for f, mod in [("launch_ridge", "run_ridge"), ("launch_tomek", "run_tomek")]:
    s = open(f + ".py").read().replace(f"import {mod} as", f"import {mod}_x as")
    open(f + "_x.py", "w").write(s)
print("ok")
