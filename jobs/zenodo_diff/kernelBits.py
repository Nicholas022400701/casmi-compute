"""kernelBits.py — load named similarity / library helpers from a kernel source file WITHOUT running it (AST pick, exec in a clean namespace).
Generic: the caller names the file; nothing from the file is printed. numba `cache=True` is dropped (no source file to cache against)."""
import ast, time
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from numba import njit, prange

WANT_FUNCS = ['CFG', '_clean', 'entropy_sim', 'cos_sim', 'search', 'neutral_mass', 'lib_window', 'clean', 'lib_sim']
WANT_NAMES = ['MASS', 'E', 'PROTON', 'H2O', 'NH4', 'FORMATE', 'ACETATE', 'ADDUCTS', 'ICE_LIB_GATE', 'EXTRA_LIB_GATE']

def load(path, funcs=WANT_FUNCS, names=WANT_NAMES):
    src = open(path).read(); tree = ast.parse(src)
    keep, seenAssign = [], set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in funcs:
            if isinstance(node, ast.FunctionDef):
                for d in node.decorator_list:
                    if isinstance(d, ast.Call): d.keywords = [k for k in d.keywords if k.arg != 'cache']
            keep.append(node)
        elif isinstance(node, ast.Assign):
            tg = [t.id for t in node.targets if isinstance(t, ast.Name)]
            hit = [t for t in tg if t in names and t not in seenAssign]
            if hit and len(tg) == len(hit): keep.append(node); seenAssign.update(hit)
    mod = ast.Module(body=keep, type_ignores=[]); ast.fix_missing_locations(mod)
    ns = dict(np=np, pd=pd, pa=pa, pq=pq, njit=njit, prange=prange, time=time)
    exec(compile(mod, '<kernelBits>', 'exec'), ns)
    missing = [n for n in funcs + names if n not in ns]
    return ns, missing
