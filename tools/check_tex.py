"""Static checks of main.tex: citations vs bibliography, number macros vs results/numbers.tex,
pending markers, brace / environment balance. Usage: python tools/check_tex.py"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
t = (ROOT / "main.tex").read_text(encoding="utf-8")
body = re.sub(r"(?<!\\)%.*", "", t)                        # strip comments
cites = {k.strip() for m in re.findall(r"\\cite[pt]?(?:\[[^\]]*\])*\{([^}]*)\}", body)
         for k in m.split(",")}
bibs = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", body))
defs = set(re.findall(r"\\newcommand\{\\(\w+)\}", (ROOT / "results/numbers.tex").read_text()))
used = set(re.findall(r"\\([a-zA-Z]+)", body))
known_prefix = r"^(n[A-Z]|prev|pct|gbif|effort|prior[A-Z]|survey|fold|block|buffer|kNeigh|tuning|res|abl)"
undefined = {u for u in used if re.match(known_prefix, u)} - defs
ok = True
for name, s in [("cited but missing from bibliography", cites - bibs),
                ("in bibliography but never cited", bibs - cites),
                ("number macros used but not generated", undefined)]:
    print(f"{name}: {sorted(s) or 'none'}")
    ok &= not s or name.startswith("in bibliography")
print("unbalanced braces:", body.count("{") - body.count("}"))
print("\\begin vs \\end:", len(re.findall(r"\\begin\{", body)), len(re.findall(r"\\end\{", body)))
print("pending markers:", len(re.findall(r"\\pending\{", body)))
sys.exit(0 if ok and body.count("{") == body.count("}") else 1)
