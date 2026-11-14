"""Compare two folders of generated LaTeX/JSON, ignoring file names.

Checks macro name -> definition, the non-macro lines of every .tex file (as a
multiset) and the leaf values of every .json file. Exits 1 on any difference
other than the allowed ones.

Usage: python compare_generated.py committed_dir regenerated_dir
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

BS = chr(92)
MACRO = re.compile(re.escape(BS) + r"newcommand\{" + re.escape(BS) + r"(R[A-Za-z]+)\}(.*)")

# Depend on the audio clips, which are not in the repository.
ALLOWED = {"RClipsIntShaOk"}
ALLOWED_JSON_KEYS = {"sha_ok"}


def load(d):
    macros, other, leaves = {}, Counter(), Counter()
    for f in sorted(Path(d).glob("*.tex")):
        for line in f.read_text(encoding="utf8").splitlines():
            m = MACRO.match(line)
            if m:
                macros[m.group(1)] = m.group(2)
            elif not line.startswith("%"):
                other[line] += 1

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k not in ALLOWED_JSON_KEYS:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            leaves[json.dumps(o)] += 1

    for f in sorted(Path(d).glob("*.json")):
        walk(json.loads(f.read_text(encoding="utf8")))
    return macros, other, leaves


(ma, oa, la), (mb, ob, lb) = load(sys.argv[1]), load(sys.argv[2])
missing = sorted(set(ma) - set(mb))
extra = sorted(set(mb) - set(ma))
diff = sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k])
print(f"macros committed={len(ma)} regenerated={len(mb)} missing={len(missing)} extra={len(extra)} differing={len(diff)}")
for k in missing:
    print("  MISSING", k)
for k in extra:
    print("  EXTRA  ", k)
for k in diff:
    print("  DIFF   ", k, "|", ma[k][:80], "|", mb[k][:80], "(allowed)" if k in ALLOWED else "")
lines_diff = sum((oa - ob).values()) + sum((ob - oa).values())
print(f"non-macro tex lines: committed={sum(oa.values())} regenerated={sum(ob.values())} differing={lines_diff}")
for line in (oa - ob).elements():
    print("  -", line[:140])
for line in (ob - oa).elements():
    print("  +", line[:140])
leaves_diff = sum((la - lb).values()) + sum((lb - la).values())
print(f"json leaves: committed={sum(la.values())} regenerated={sum(lb.values())} differing={leaves_diff}")
for x in (la - lb).elements():
    print("  -", x[:140])
for x in (lb - la).elements():
    print("  +", x[:140])

bad = missing or extra or [k for k in diff if k not in ALLOWED] or lines_diff or leaves_diff
print("RESULT:", "MISMATCH" if bad else "OK")
sys.exit(1 if bad else 0)
