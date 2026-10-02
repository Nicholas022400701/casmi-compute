#!/usr/bin/env python3
"""zenodoDiff.py — LABEL-FREE COUNT: an open spectral-library release (parquet) vs the competition train set.
usage: zenodoDiff.py --zen full.parquet [--zenFiltered filtered.parquet] --train train.parquet --test test.parquet --bundle <dir> --cell <id> --out <dir>
                     [--nControl 3] [--seed 0] [--trainRowGroups 0=all] [--maxPeaks 0=all]
What it writes (out/): report.json (counts, quantiles, histograms, md5s), top50.csv (molecule_id + scores only), Z_library.parquet (release rows of the
structures absent from train, all columns), Z_trainlike.parquet (same rows, train.parquet column names), Z_keys.txt, README_LICENSE.txt.
Log rule (public runner): counts and timings only — no identifiers, no structures."""
import argparse, hashlib, json, os, sys, time, collections
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import kernelBits
T0 = time.time()
def log(*a): print(f'[{time.strftime("%H:%M:%S")} +{time.time()-T0:.0f}s]', *a, flush=True)
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
def q(v, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return None if v.size == 0 else {str(p): round(float(np.quantile(v, p)), 4) for p in qs} | {'n': int(v.size), 'mean': round(float(v.mean()), 4)}
def hist(v, edges):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; c, _ = np.histogram(v, bins=edges)
    return {f'{edges[i]:.2f}-{edges[i+1]:.2f}': int(c[i]) for i in range(len(c))}
ap = argparse.ArgumentParser(); ap.add_argument('--zen', required=True); ap.add_argument('--zenFiltered', default=''); ap.add_argument('--train', required=True); ap.add_argument('--test', required=True)
ap.add_argument('--bundle', required=True); ap.add_argument('--cell', required=True); ap.add_argument('--out', required=True); ap.add_argument('--nControl', type=int, default=3); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--trainRowGroups', type=int, default=0); ap.add_argument('--maxPeaks', type=int, default=0); ap.add_argument('--maxZ', type=int, default=0, help='debug: cap Z to the first N keys')
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True); R = dict(args={k: (v if k != 'bundle' else '<bundle>') for k, v in vars(A).items()}, versions=dict(numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__))

# ---- 1. kernel helpers (by name, from the private bundle; nothing printed) ----
cells = json.load(open(os.path.join(A.bundle, 'cells.json'))); base = cells[A.cell]['RP_BASE']
ns, missing = kernelBits.load(os.path.join(A.bundle, base)); assert not missing, f'kernel helpers missing: {missing}'
CFG, clean, search, lib_window, lib_sim, neutral_mass, entropy_sim = ns['CFG'], ns['clean'], ns['search'], ns['lib_window'], ns['lib_sim'], ns['neutral_mass'], ns['entropy_sim']
GATE = float(ns['ICE_LIB_GATE']); R['kernel'] = dict(baseMd5=md5(os.path.join(A.bundle, base))[:8], ppmWin=CFG.PPM_WIN, mzTol=CFG.MZ_TOL, intFloor=CFG.INT_FLOOR, maxPeaks=CFG.MAX_PEAKS, intPower=CFG.INT_POWER, entWeight=bool(CFG.ENT_WEIGHT), gate=GATE, nAdducts=len(ns['ADDUCTS']))
log(f'kernel helpers loaded: gate {GATE}, window {CFG.PPM_WIN} ppm, tol {CFG.MZ_TOL}')

# ---- 2. train keys + the kernel-style library over train rows ----
def build_lib(tab, keyCol, smiCol, addCol, precCol, mzCol, itCol, extra=None):
    mzc = tab.column(mzCol).combine_chunks(); itc = tab.column(itCol).combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64); allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32); allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    prec = tab.column(precCol).to_numpy(zero_copy_only=False).astype(np.float64); add = np.asarray(tab.column(addCol).cast(pa.string()).to_pylist(), dtype=object)
    ik = np.asarray(tab.column(keyCol).cast(pa.string()).to_pylist(), dtype=object)
    nm = neutral_mass(prec, add); ok = np.isfinite(nm); order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')
    L = dict(off=off, mz=allmz, it=allin, nm=nm, ik=ik, order=order, snm=nm[order], n_ok=int(ok.sum()), npk=np.diff(off))
    if extra: L.update({k: np.asarray(tab.column(c).cast(pa.string()).to_pylist(), dtype=object) for k, c in extra.items()})
    return L
tf = pq.ParquetFile(A.train); rgs = list(range(tf.metadata.num_row_groups if not A.trainRowGroups else min(A.trainRowGroups, tf.metadata.num_row_groups)))
tk = tf.read_row_groups(rgs, columns=['inchikey14', 'ingest_lib', 'adduct'])
trainKeys = set(tk.column('inchikey14').to_pylist()); e180mask = pc.equal(tk.column('ingest_lib'), 'enveda-180'); e180Keys = set(pc.filter(tk.column('inchikey14'), e180mask).to_pylist())
e180AdductCounts = collections.Counter(pc.filter(tk.column('adduct'), e180mask).to_pylist()); keptAdducts = {a for a in e180AdductCounts if a in ns['ADDUCTS']}
e180RowsPerKey = pd.Series(pc.filter(tk.column('inchikey14'), e180mask).to_pylist()).value_counts()
R['train'] = dict(rows=tk.num_rows, rowGroups=len(rgs), keys=len(trainKeys), e180Rows=int(pc.sum(e180mask).as_py()), e180Keys=len(e180Keys), libs=dict(collections.Counter(tk.column('ingest_lib').to_pylist()).most_common()),
                  e180Adducts=e180AdductCounts.most_common(), keptAdductsInKernelTable=sorted(keptAdducts))
log(f'train: {tk.num_rows:,} rows, {len(trainKeys):,} keys, e180 keys {len(e180Keys):,}')
tl = tf.read_row_groups(rgs, columns=['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'ingest_lib'])
Ltr = build_lib(tl, 'inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', extra=dict(lib='ingest_lib')); del tl
R['train'].update(libSpectra=int(len(Ltr['off']) - 1), libFiniteNm=Ltr['n_ok']); log(f'train library: {len(Ltr["off"])-1:,} spectra, finite nm {Ltr["n_ok"]:,}')

# ---- 3. the release: schema sniffing ----
def sniff(pf):
    names = pf.schema_arrow.names; types = {n: pf.schema_arrow.field(n).type for n in names}; low = {n: n.lower() for n in names}
    def pick(pred, prefer=()):
        for p in prefer:
            if p in names: return p
        c = [n for n in names if pred(low[n])]; return c[0] if c else None
    key = pick(lambda s: 'inchikey' in s, ('inchikey', 'InChIKey', 'inchi_key'))
    smi = pick(lambda s: 'smiles' in s, ('smiles', 'normalized_smiles', 'canonical_smiles'))
    prec = pick(lambda s: ('precursor' in s and ('mz' in s or 'm/z' in s)) or s == 'pepmass', ('precursor_mz', 'pepmass'))
    add = pick(lambda s: s in ('adduct', 'precursor_type', 'adduct_type') or s.startswith('adduct'), ('adduct',))
    pol = pick(lambda s: s in ('ionization_mode', 'polarity', 'ion_mode', 'ionmode') or 'polarity' in s or 'ion_mode' in s or 'ionization' in s, ('ionmode', 'ionization_mode'))
    ce = pick(lambda s: 'collision' in s or s in ('ce', 'nce'), ('collision_energy_ev', 'collision_energies', 'collision_energy'))
    formula = pick(lambda s: s in ('formula', 'molecular_formula'), ('formula', 'molecular_formula'))
    title = pick(lambda s: s in ('title', 'spectrum_id', 'id'), ('title',))
    lists = [n for n in names if pa.types.is_list(types[n]) or pa.types.is_large_list(types[n])]
    mzL = [n for n in lists if 'mz' in low[n] or 'mass' in low[n]]; itL = [n for n in lists if 'intens' in low[n] or low[n].endswith('_int') or 'abundance' in low[n]]
    peaks = None
    if mzL and itL: peaks = ('lists', mzL[0], itL[0])
    else:
        st = [n for n in lists if pa.types.is_struct(types[n].value_type)]
        if st: peaks = ('structs', st[0], None)
        else:
            s = pick(lambda s: s in ('peaks', 'spectrum', 'ms2_peaks', 'peaks_json') or 'peak' in s); peaks = ('string', s, None) if s else None
    return dict(key=key, smiles=smi, prec=prec, adduct=add, pol=pol, ce=ce, formula=formula, title=title, peaks=peaks, names=names, types={n: str(types[n]) for n in names})
zf = pq.ParquetFile(A.zen); S = sniff(zf); R['release'] = dict(file=os.path.basename(A.zen), bytes=os.path.getsize(A.zen), rows=zf.metadata.num_rows, rowGroups=zf.metadata.num_row_groups, columns=S['types'], picked={k: v for k, v in S.items() if k not in ('names', 'types')})
log(f'release: {zf.metadata.num_rows:,} rows, {len(S["names"])} columns, {zf.metadata.num_row_groups} row groups'); assert S['key'] and S['prec'] and S['peaks'], f'schema not understood: {S}'

def peaks_of(tab, S):
    """-> (offsets int64, mz float32, it float32) for the rows of tab, from whichever peak layout the file uses."""
    kind, c1, c2 = S['peaks']
    if kind == 'lists':
        mzc = tab.column(c1).combine_chunks(); itc = tab.column(c2).combine_chunks()
        if not pa.types.is_floating(mzc.type.value_type): mzc = mzc.cast(pa.list_(pa.float64()))
        if not pa.types.is_floating(itc.type.value_type): itc = itc.cast(pa.list_(pa.float64()))
        return mzc.offsets.to_numpy().astype(np.int64), mzc.values.to_numpy(zero_copy_only=False).astype(np.float32), itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    rows = tab.column(c1).to_pylist(); off = [0]; mz = []; it = []
    for r in rows:
        if kind == 'structs':
            for d in (r or []):
                v = list(d.values()); mz.append(float(v[0])); it.append(float(v[1]))
        else:
            if r:
                for tok in str(r).replace('\n', ' ').replace(';', ' ').replace(',', ' ').replace(':', ' ').split():
                    try: mz.append(float(tok)) if len(mz) == len(it) else it.append(float(tok))
                    except ValueError: pass
                if len(mz) != len(it): mz = mz[:len(it)]
        off.append(len(mz))
    return np.asarray(off, np.int64), np.asarray(mz, np.float32), np.asarray(it, np.float32)

MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.99491461956, S=31.97207100, P=30.97376163, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473, Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213, Li=7.01600455, Mg=23.9850417, Ca=39.96259098, Fe=55.9349375, Zn=63.9291422, Cu=62.9295975, Mn=54.9380451, Co=58.9331950, Ni=57.9353429, Al=26.98153863, As=74.9215965, Sn=119.9021947, Ag=106.905097, Au=196.9665687, Pt=194.9647911, Hg=201.970643, Pb=207.9766521, Ti=47.9479463, Cr=51.9405075, Ba=137.9052472, Sr=87.9056121, Cs=132.905451933, Rb=84.911789738, Ge=73.9211778, Te=129.9062244, Sb=120.9038157, Bi=208.9803987, Ga=68.9255736, V=50.9439595, W=183.9509312, Mo=97.9054082, Zr=89.9047044, Pd=105.903486, Ru=101.9043493, Rh=102.905504, D=2.0141017778)
_FRE = __import__('re').compile(r'([A-Z][a-z]?)(\d*)')
def formula_mass(f):
    if not isinstance(f, str) or not f: return np.nan
    f = f.split('+')[0].split('-')[0].strip(); m = 0.0; seen = 0
    for el, n in _FRE.findall(f):
        if el not in MONO: return np.nan
        m += MONO[el] * (int(n) if n else 1); seen += 1
    return m if seen else np.nan
def to_float(col):
    try: return col.cast(pa.float64()).to_numpy(zero_copy_only=False)
    except Exception:
        out = np.full(len(col), np.nan)
        for i, v in enumerate(col.to_pylist()):
            try: out[i] = float(str(v).split()[0])
            except Exception: pass
        return out
# ---- 4. pass 1: metadata of every release row (key, precursor, adduct, polarity, CE, nPeaks, every scalar column) ----
BIG = {S['smiles'], 'iupac_name', 'enamine_catalog_id', 'pubchem_cid', 'adduct_formula', 'ccs_species'}
scalarCols = [n for n in S['names'] if not (pa.types.is_list(zf.schema_arrow.field(n).type) or pa.types.is_large_list(zf.schema_arrow.field(n).type) or pa.types.is_struct(zf.schema_arrow.field(n).type)) and n not in BIG]
listCol = S['peaks'][1] if S['peaks'][0] in ('lists', 'structs') else None
numCols, catCols = [], []
parts = []
for b in zf.iter_batches(batch_size=200_000, columns=scalarCols + ([listCol] if (listCol and 'num_peaks' not in scalarCols) else [])):
    d = {}
    for c in scalarCols:
        col = b.column(c)
        if c == S['key']: d['ik14'] = pd.Series([x[:14] if x else '' for x in col.to_pylist()], dtype='string'); continue
        if c == S['title']:
            d['titleIdx'] = pd.to_numeric(pd.Series(col.to_pylist()).astype(str).str.extract(r'(\d+)')[0], errors='coerce').astype('float64'); continue
        if pa.types.is_floating(col.type) or pa.types.is_integer(col.type): d[c] = col.to_numpy(zero_copy_only=False).astype('float64'); numCols.append(c); continue
        v = pd.Series(col.to_pylist(), dtype='object')
        f = pd.to_numeric(v, errors='coerce')
        if f.notna().mean() > 0.9: d[c] = f.astype('float32'); numCols.append(c)
        else: d[c] = v.astype('category'); catCols.append(c)
    d = pd.DataFrame(d)
    d['nPeaks'] = d['num_peaks'] if 'num_peaks' in d else (pc.list_value_length(b.column(listCol)).to_numpy(zero_copy_only=False) if listCol else -1)
    parts.append(d)
M = pd.concat(parts, ignore_index=True); del parts; numCols = sorted(set(numCols)); catCols = sorted(set(catCols))
for c in catCols: M[c] = M[c].astype('string')
def polarity_of(addStr, modeStr=None):
    """'+' / '-' per row from the adduct string's trailing sign, else from an ion-mode string; '' when unknown."""
    out = np.array([(a.strip()[-1] if isinstance(a, str) and a.strip() and a.strip()[-1] in '+-' else '') for a in addStr], dtype=object)
    if modeStr is not None:
        m = np.array([('-' if isinstance(x, str) and 'neg' in x.lower() else '+' if isinstance(x, str) and 'pos' in x.lower() else '') for x in modeStr], dtype=object); out = np.where(out == '', m, out)
    return out
def implied_adduct(pep, nmF, pol):
    """kernel adduct name (same polarity) whose neutral mass from pep agrees with the formula mass within the kernel window (else None); vectorised."""
    out = np.full(len(pep), None, dtype=object); done = np.zeros(len(pep), bool)
    for a, (n, z, d) in ns['ADDUCTS'].items():
        nm = (pep * z - d) / n; hit = (~done) & (pol == a[-1]) & np.isfinite(nmF) & np.isfinite(nm) & (np.abs(nm - nmF) <= nmF * CFG.PPM_WIN / 1e6)
        out[hit] = a; done |= hit
    return out
M['pep'] = M[S['prec']].astype(float).values if S['prec'] in M else np.nan
fmass = {f: formula_mass(f) for f in M[S['formula']].dropna().unique()} if S['formula'] and S['formula'] in M else {}
M['nmF'] = M[S['formula']].map(fmass).astype(float) if fmass else np.nan
M['pol'] = polarity_of(M[S['adduct']].astype(object).values if S['adduct'] in M else [None] * len(M), M[S['pol']].astype(object).values if (S['pol'] and S['pol'] in M) else None)
M['kAdduct'] = implied_adduct(M['pep'].values.astype(float), M['nmF'].values.astype(float), M['pol'].values); M['kept'] = M['kAdduct'].isin(keptAdducts)
R['adductMap'] = dict(polarity=collections.Counter(M['pol'].tolist()).most_common(), rowsWithFormulaMass=int(M['nmF'].notna().sum()), rowsWithKernelAdduct=int(M['kAdduct'].notna().sum()), rowsKeptAdduct=int(M['kept'].sum()),
                      releaseAdductToKernel=[(str(k[0]), str(k[1]), int(v)) for k, v in M.groupby([M[S['adduct']].astype(str), M['kAdduct'].astype(str)]).size().sort_values(ascending=False).head(40).items()] if S['adduct'] in M else None)
log(f'adducts: {R["adductMap"]["rowsWithKernelAdduct"]:,} rows map to a kernel adduct, {R["adductMap"]["rowsKeptAdduct"]:,} to a host-kept one')
M['inTrain'] = M['ik14'].isin(trainKeys); M['inE180'] = M['ik14'].isin(e180Keys)
if 'titleIdx' in M and M['titleIdx'].notna().any(): numCols.append('titleIdx')
relKeys = set(M['ik14']); Z = sorted(relKeys - trainKeys); Zfull = len(Z)
if A.maxZ: Z = Z[:A.maxZ]
Zset = set(Z)
R['diff'] = dict(releaseKeys=len(relKeys), releaseRows=len(M), Z_keys=Zfull, Z_keysUsed=len(Z), Z_rows=int((~M['inTrain']).sum()), keysInTrain=len(relKeys & trainKeys), keysInE180=len(relKeys & e180Keys),
                 e180KeysNotInRelease=len(e180Keys - relKeys), trainKeysNotInRelease=len(trainKeys - relKeys), e180RowsVsReleaseRowsOfTrainKeys=[R['train']['e180Rows'], int(M['inTrain'].sum())])
log(f'diff: release keys {len(relKeys):,} / rows {len(M):,}; Z = {len(Z):,} keys / {R["diff"]["Z_rows"]:,} rows; e180 keys not in release {R["diff"]["e180KeysNotInRelease"]:,}')
kp = M[M['kept']]; keptPerKey = kp.groupby('ik14').size(); keysInTrainRel = sorted(relKeys & trainKeys)
cmp = pd.DataFrame(dict(rel=keptPerKey.reindex(keysInTrainRel).fillna(0).astype(int).values, tr=e180RowsPerKey.reindex(keysInTrainRel).fillna(0).astype(int).values), index=keysInTrainRel)
Zk = sorted(set(keptPerKey.index) & Zset)
R['adductFilter'] = dict(releaseRowsKept=int(len(kp)), trainE180Rows=R['train']['e180Rows'], releaseKeptRowsOfTrainKeys=int(cmp.rel.sum()), trainE180RowsOfThoseKeys=int(cmp.tr.sum()),
                         keysEqualRows=int((cmp.rel == cmp.tr).sum()), keysReleaseMore=int((cmp.rel > cmp.tr).sum()), keysTrainMore=int((cmp.rel < cmp.tr).sum()), nKeysCompared=int(len(cmp)), diffQ=q((cmp.rel - cmp.tr).values),
                         keptRowsByAdduct_release=collections.Counter(kp['kAdduct'].tolist()).most_common(), Z_keysWithKeptRow=len(Zk), Z_keptRows=int(kp['ik14'].isin(Zset).sum()), Z_keysKeptOnlyFragmentAdducts=len(Zset) - len(Zk))
log(f'adduct filter: release kept rows {len(kp):,} vs train e180 rows {R["train"]["e180Rows"]:,}; Z keys with a kept-adduct spectrum {len(Zk):,} of {len(Zset):,}')
if A.zenFiltered and os.path.exists(A.zenFiltered):
    ff = pq.ParquetFile(A.zenFiltered); Sf = sniff(ff); fk = set()
    nF = 0
    for b in ff.iter_batches(batch_size=500_000, columns=[Sf['key']]): fk.update(s[:14] for s in b.column(0).to_pylist() if s); nF += b.num_rows
    R['filtered'] = dict(file=os.path.basename(A.zenFiltered), rows=nF, keys=len(fk), keysInTrain=len(fk & trainKeys), keysNotInTrain=len(fk - trainKeys), Z_inFiltered=len(Zset & fk), e180KeysNotInFiltered=len(e180Keys - fk), fullMinusFiltered=len(relKeys - fk), trainKeysNotInFiltered=len(trainKeys - fk))
    log(f'filtered variant: {nF:,} rows, {len(fk):,} keys; Z in filtered {len(Zset & fk):,}; e180 keys not in filtered {len(e180Keys - fk):,}')
    M['inFiltered'] = M['ik14'].isin(fk)

# ---- 5. profiles: Z rows vs the rest of the release (every scalar column), spectra per compound, polarity / adduct / CE ----
def profile(d):
    out = dict(rows=int(len(d)), keys=int(d['ik14'].nunique()), spectraPerKey=q(d.groupby('ik14').size().values), nPeaks=q(d['nPeaks'].values.astype(float)))
    for c in numCols:
        if c in d: out[f'num:{c}'] = q(d[c].values.astype(float))
    for c in catCols:
        vc = d[c].astype(str).value_counts(); out[f'cat:{c}'] = dict(nUnique=int(vc.size), top=[(k[:40], int(v)) for k, v in vc.head(12).items()])
    if 'titleIdx' in d and M['titleIdx'].notna().any():
        lo, hi = float(M['titleIdx'].min()), float(M['titleIdx'].max()); out['positionInFileDeciles'] = hist((d['titleIdx'].values - lo) / max(hi - lo, 1), np.linspace(0, 1, 11))
    return out
zm = M['ik14'].isin(Zset); zkm = M['ik14'].isin(set(Zk)) & M['kept']; R['profile'] = dict(Z=profile(M[zm]), Zkept_keptRows=profile(M[zkm]) if zkm.any() else None, rest=profile(M[~zm].sample(min(300_000, int((~zm).sum())), random_state=0)), restKeptRows=profile(M[(~zm) & M['kept']].sample(min(300_000, int(((~zm) & M['kept']).sum())), random_state=0)))
if 'inFiltered' in M: R['profile']['Z_filteredShare'] = round(float(M.loc[zm, 'inFiltered'].mean()), 4); R['profile']['rest_filteredShare'] = round(float(M.loc[~zm, 'inFiltered'].mean()), 4)
log('profiles done')

# ---- 6. pass 2: the spectra of Z (library) and of the control draws (train keys present in the release) ----
rng = np.random.default_rng(A.seed); poolCtl = sorted(relKeys & trainKeys)
# the visible test file's molecules are train copies -> resolve their keys by exact spectrum identity so that the controls never contain them
te = pq.read_table(A.test).to_pandas(); teKeys = set()
precIdx = collections.defaultdict(list)
for i, (p, k) in enumerate(zip(np.round(tl_prec := tf.read_row_groups(rgs, columns=['precursor_mz']).column(0).to_numpy(zero_copy_only=False), 4), Ltr['ik'])): precIdx[p].append(i)
dummyResolved = 0
for r in te.itertuples():
    p = round(float(r.precursor_mz), 4); qmz = np.asarray(r.ms2_mzs, np.float32)
    for i in precIdx.get(p, []):
        a, b = Ltr['off'][i], Ltr['off'][i + 1]
        if b - a == len(qmz) and (len(qmz) == 0 or np.allclose(Ltr['mz'][a:b][:5], qmz[:5], atol=1e-4)): teKeys.add(Ltr['ik'][i]); dummyResolved += 1; break
del precIdx, tl_prec
R['test'] = dict(rows=int(len(te)), molecules=int(te['molecule_id'].nunique()), spectraResolvedToTrain=dummyResolved, keysResolved=len(teKeys))
poolCtl = [k for k in poolCtl if k not in teKeys]
ctlSets = [set(rng.choice(poolCtl, size=min(len(Z), len(poolCtl)), replace=False).tolist()) for _ in range(A.nControl)]
want = Zset.union(*ctlSets); keep = []
for b in zf.iter_batches(batch_size=100_000):
    k = pa.array([s[:14] if s else '' for s in b.column(S['key']).to_pylist()]); m = pc.is_in(k, value_set=pa.array(sorted(want)))
    if pc.any(m).as_py(): keep.append(pa.Table.from_batches([b]).filter(m))
ZT = pa.concat_tables(keep); del keep
zk = np.asarray([s[:14] for s in ZT.column(S['key']).to_pylist()], dtype=object); isZ = np.isin(zk, Z)
log(f'pass 2: {ZT.num_rows:,} rows kept ({int(isZ.sum()):,} Z rows, {A.nControl} control draws of {len(ctlSets[0]):,} keys)')

def lib_from(tab, keys14):
    off, mz, it = peaks_of(tab, S); prec = to_float(tab.column(S['prec']))
    add = np.asarray(tab.column(S['adduct']).cast(pa.string()).to_pylist(), dtype=object) if S['adduct'] else np.asarray(['[M+H]+'] * tab.num_rows, dtype=object)
    nmF = np.array([formula_mass(f) for f in tab.column(S['formula']).to_pylist()]) if S['formula'] else np.full(tab.num_rows, np.nan)
    pol = polarity_of(add, np.asarray(tab.column(S['pol']).cast(pa.string()).to_pylist(), dtype=object) if S['pol'] else None); kAdd = implied_adduct(prec, nmF, pol); hasK = np.array([a is not None for a in kAdd]); kAddS = np.array([a if a is not None else '' for a in kAdd], dtype=object)
    nmK = neutral_mass(prec, kAddS); nm = np.where(np.isfinite(nmK), nmK, nmF); ok = np.isfinite(nm); order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')
    kept = np.array([a in keptAdducts for a in kAdd])
    return dict(off=off, mz=mz, it=it, nm=nm, ik=np.asarray(keys14, dtype=object), order=order, snm=nm[order], n_ok=int(ok.sum()), npk=np.diff(off), add=add, kAdd=kAddS, kept=kept, prec=prec,
                nmSource=dict(kernelAdduct=int(np.isfinite(nmK).sum()), formulaOnly=int((~np.isfinite(nmK) & np.isfinite(nmF)).sum()), none=int((~ok).sum()), keptAdductRows=int(kept.sum())))
ZTz = ZT.filter(pa.array(isZ)); LZ = lib_from(ZTz, zk[isZ]); R['Z'] = dict(spectra=ZTz.num_rows, keys=int(len(set(zk[isZ]))), finiteNm=LZ['n_ok'], nmSource=LZ['nmSource'], adductTop=collections.Counter(LZ['add'].tolist()).most_common(12), keptRows=int(LZ['kept'].sum()), keysWithKeptRow=int(len(set(LZ['ik'][LZ['kept']]))))
log(f'Z library: {ZTz.num_rows:,} spectra, finite nm {LZ["n_ok"]:,}')

# ---- 7. Z spectra vs the TRAIN library (kernel recipe): duplicates (>= 0.99), gate-level matches; per spectrum and per structure ----
def best_against(L, qmz, qit, target):
    cand = lib_window(L, target, target * CFG.PPM_WIN / 1e6)
    if len(cand) == 0: return -1.0, -1, 0
    qm, qp = clean(qmz, qit)
    if len(qm) == 0: return -2.0, -1, len(cand)
    sc = search(qm, qp, cand, L['off'], L['mz'], L['it'], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT, 1)
    j = int(np.argmax(sc)); return float(sc[j]), int(cand[j]), len(cand)
bestZ = np.full(ZTz.num_rows, np.nan); bestLib = []; nCand = np.zeros(ZTz.num_rows, int); t = time.time()
for i in range(ZTz.num_rows):
    if not np.isfinite(LZ['nm'][i]): bestLib.append(None); continue
    a, b = LZ['off'][i], LZ['off'][i + 1]; s, j, nc = best_against(Ltr, LZ['mz'][a:b], LZ['it'][a:b], float(LZ['nm'][i])); bestZ[i] = s; nCand[i] = nc; bestLib.append(Ltr['lib'][j] if j >= 0 else None)
perKey = pd.DataFrame(dict(k=LZ['ik'], s=bestZ)).groupby('k').s.max()
R['Z_vs_train'] = dict(spectraScored=int(np.isfinite(bestZ).sum()), noWindow=int((bestZ == -1).sum()), emptyClean=int((bestZ == -2).sum()), candidatesPerSpectrum=q(nCand[nCand > 0]),
                       spectraGe099=int((bestZ >= 0.99).sum()), spectraGe095=int((bestZ >= 0.95).sum()), spectraGeGate=int((bestZ >= GATE).sum()), hist=hist(bestZ[bestZ >= 0], np.linspace(0, 1, 21)),
                       keysGe099=int((perKey >= 0.99).sum()), keysGeGate=int((perKey >= GATE).sum()), keysScored=int(perKey.notna().sum()), keptRows=dict(n=int(LZ['kept'].sum()), ge099=int((bestZ[LZ['kept']] >= 0.99).sum()), geGate=int((bestZ[LZ['kept']] >= GATE).sum()), hist=hist(bestZ[LZ['kept'] & (bestZ >= 0)], np.linspace(0, 1, 21))), bestLibOfGe099=collections.Counter([l for s, l in zip(bestZ, bestLib) if s >= 0.99]).most_common(8), sec=round(time.time() - t, 1))
log(f'Z vs train library: spectra >= 0.99 {R["Z_vs_train"]["spectraGe099"]:,}, >= gate {R["Z_vs_train"]["spectraGeGate"]:,}; keys >= gate {R["Z_vs_train"]["keysGeGate"]:,} of {len(perKey):,} ({time.time()-t:.0f}s)')
# kernel-style per structure: all spectra of the structure fused at the median nm (the kernel's per-molecule rule)
t = time.time(); lvKey = {}
for k, idx in pd.Series(np.arange(ZTz.num_rows)).groupby(LZ['ik']).groups.items():
    idx = np.asarray(list(idx)); nms = LZ['nm'][idx]; nms = nms[np.isfinite(nms)]
    if nms.size == 0: continue
    specs = [(LZ['mz'][LZ['off'][i]:LZ['off'][i+1]], LZ['it'][LZ['off'][i]:LZ['off'][i+1]]) for i in idx]
    hits = lib_sim(Ltr, specs, float(np.median(nms))); lvKey[k] = max(hits.values()) if hits else 0.0
lv = np.array(list(lvKey.values())); R['Z_vs_train']['fusedKeys'] = dict(n=int(lv.size), lvMaxGeGate=int((lv >= GATE).sum()), lvMaxGe099=int((lv >= 0.99).sum()), hist=hist(lv, np.linspace(0, 1, 21)), sec=round(time.time() - t, 1))

# ---- 8. RDKit standardisation collapse of Z structures into train keys ----
try:
    from rdkit import Chem, RDLogger; from rdkit.Chem.MolStandardize import rdMolStandardize; RDLogger.DisableLog('rdApp.*')
    smiCol = S['smiles']; firstSmi = {}
    for k, s in zip(zk[isZ], ZTz.column(smiCol).to_pylist() if smiCol else [None] * ZTz.num_rows):
        if k not in firstSmi and s: firstSmi[k] = s
    lf = rdMolStandardize.LargestFragmentChooser(); un = rdMolStandardize.Uncharger(); c = collections.Counter()
    for k, s in firstSmi.items():
        m = Chem.MolFromSmiles(s)
        if m is None: c['parseFail'] += 1; continue
        k0 = Chem.MolToInchiKey(m)[:14]; c['rawKeyEqualsFile'] += int(k0 == k); c['rawKeyInTrain'] += int(k0 in trainKeys)
        m2 = un.uncharge(lf.choose(m)); k2 = Chem.MolToInchiKey(m2)[:14]; c['stdKeyInTrain'] += int(k2 in trainKeys); c['stdKeyChanged'] += int(k2 != k0)
        try:
            from rdkit.Chem import rdMolDescriptors; c['hasMetal'] += int(any(a.GetSymbol() in ('Na', 'K', 'Li', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu', 'Pt', 'Mn', 'Co', 'Ni', 'Al', 'Ag', 'Au', 'Hg', 'Pb', 'Sn', 'Ti', 'Cr', 'Ba', 'Sr', 'Cs', 'Rb') for a in m.GetAtoms()))
            c['multiFragment'] += int('.' in s)
        except Exception: pass
    R['Z_standardise'] = dict(keysWithSmiles=len(firstSmi), **c); log(f'standardise: std key in train {c["stdKeyInTrain"]:,} of {len(firstSmi):,} (raw key in train {c["rawKeyInTrain"]:,})')
except Exception as e:
    R['Z_standardise'] = dict(error=repr(e)[:200]); log('standardise: skipped')

# ---- 9. the visible test file vs Z (treatment) and vs the control draws — the kernel's lib gate, per molecule (fused) and per spectrum ----
te['nm'] = neutral_mass(te['precursor_mz'].to_numpy(np.float64), np.asarray(te['adduct'].astype(str).tolist(), dtype=object))
def match_test(L):
    perMol = []; perSpec = []
    for mid, sub in te.groupby('molecule_id', sort=False):
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if nms.size == 0: perMol.append((mid, np.nan, np.nan, 0)); continue
        target = float(np.median(nms)); specs = [(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32)) for r in sub.itertuples()]
        hits = lib_sim(L, specs, target); bestK = max(hits, key=hits.get) if hits else None; best = hits[bestK] if hits else 0.0
        dppm = np.nan
        if bestK is not None:
            cand = lib_window(L, target, target * CFG.PPM_WIN / 1e6); nmK = L['nm'][cand][L['ik'][cand] == bestK]; dppm = float(np.min(np.abs(nmK - target)) / target * 1e6) if nmK.size else np.nan
        perMol.append((mid, best, dppm, int(sum(len(s[0]) for s in specs) / len(specs))))
        for r in sub.itertuples():
            s, j, nc = best_against(L, np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32), target); perSpec.append(max(s, 0.0))
    pm = pd.DataFrame(perMol, columns=['molecule_id', 'best', 'dppm', 'nPeaksMean']); ps = np.asarray(perSpec)
    return pm, dict(molecules=int(len(pm)), moleculesGeGate=int((pm.best >= GATE).sum()), moleculesGe099=int((pm.best >= 0.99).sum()), moleculesGe05=int((pm.best >= 0.5).sum()), spectra=int(ps.size), spectraGeGate=int((ps >= GATE).sum()), spectraGe099=int((ps >= 0.99).sum()), histMolecules=hist(pm.best.fillna(0).values, np.linspace(0, 1, 21)), bestQ=q(pm.best.values))
t = time.time(); pmZ, R['test_vs_Z'] = match_test(LZ); R['test_vs_Z']['sec'] = round(time.time() - t, 1)
if LZ['kept'].any():
    LZk = lib_from(ZTz.filter(pa.array(LZ['kept'])), zk[isZ][LZ['kept']]); _, R['test_vs_Zkept'] = match_test(LZk); R['test_vs_Zkept'].update(libSpectra=int(len(LZk['off']) - 1), libKeys=int(len(set(LZk['ik']))))
log(f'test vs Z: molecules >= gate {R["test_vs_Z"]["moleculesGeGate"]} of {R["test_vs_Z"]["molecules"]}, spectra >= gate {R["test_vs_Z"]["spectraGeGate"]} of {R["test_vs_Z"]["spectra"]}')
R['test_vs_control'] = []
for ci, cs in enumerate(ctlSets):
    m = np.isin(zk, sorted(cs)); Lc = lib_from(ZT.filter(pa.array(m)), zk[m]); t = time.time(); _, rc = match_test(Lc); rc.update(draw=ci, libSpectra=int(len(Lc['off']) - 1), libKeys=len(cs), sec=round(time.time() - t, 1)); R['test_vs_control'].append(rc)
    log(f'test vs control {ci}: molecules >= gate {rc["moleculesGeGate"]}, spectra >= gate {rc["spectraGeGate"]} (lib {rc["libSpectra"]:,} spectra)')
R['test_vs_control_mean'] = {k: round(float(np.mean([r[k] for r in R['test_vs_control']])), 2) for k in ('moleculesGeGate', 'moleculesGe099', 'moleculesGe05', 'spectraGeGate', 'spectraGe099')} if R['test_vs_control'] else None
top = pmZ.sort_values('best', ascending=False).head(50).copy(); top['best'] = top['best'].round(4); top['dppm'] = top['dppm'].round(2); top.to_csv(os.path.join(A.out, 'top50.csv'), index=False)

# ---- 10. outputs: Z library (all columns + train-like), keys, licence note, report ----
pq.write_table(ZTz, os.path.join(A.out, 'Z_library.parquet'), compression='zstd')
# kernel-ready file: ONLY rows whose adduct maps into the kernel vocabulary (same polarity, formula-consistent), the six library columns FIRST with train.parquet's exact dtypes
# (string / double / list<element: double>), intensities base-peak-normalised like train (raw max kept in base_peak_intensity), then provenance columns
kmask = np.array([bool(k) for k in LZ['kAdd']]); sel = np.flatnonzero(kmask); LST = pa.list_(pa.field('element', pa.float64()))
def rows_lists(idx, vals, norm=False):
    out, off = [], [0]
    for i in idx:
        a, b = LZ['off'][i], LZ['off'][i + 1]; v = vals[a:b].astype(np.float64)
        if norm and v.size and v.max() > 0: v = v / v.max()
        out.append(v); off.append(off[-1] + v.size)
    return pa.ListArray.from_arrays(pa.array(np.asarray(off, np.int32)), pa.array(np.concatenate(out) if out else np.zeros(0))).cast(LST)
bpi = np.array([float(LZ['it'][LZ['off'][i]:LZ['off'][i + 1]].max()) if LZ['off'][i + 1] > LZ['off'][i] else np.nan for i in sel])
smiAll = ZTz.column(S['smiles']).cast(pa.string()).to_pylist() if S['smiles'] else [None] * ZTz.num_rows
cols = {'inchikey14': pa.array([LZ['ik'][i] for i in sel], pa.string()), 'normalized_smiles': pa.array([smiAll[i] for i in sel], pa.string()), 'adduct': pa.array([LZ['kAdd'][i] for i in sel], pa.string()),
        'precursor_mz': pa.array(LZ['prec'][sel].astype(np.float64), pa.float64()), 'ms2_mzs': rows_lists(sel, LZ['mz']), 'ms2_normalized_intensities': rows_lists(sel, LZ['it'], norm=True),
        'base_peak_intensity': pa.array(bpi, pa.float64()), 'num_peaks': pa.array(LZ['npk'][sel].astype(np.int64)), 'ingest_lib': pa.array(['enveda-180-zenodo'] * len(sel), pa.string()),
        'inchikey': pa.array([ZTz.column(S['key']).cast(pa.string()).to_pylist()[i] for i in sel], pa.string()), 'adduct_orig': pa.array([LZ['add'][i] for i in sel], pa.string()), 'hostKeptAdduct': pa.array([bool(LZ['kept'][i]) for i in sel]),
        'neutral_mass': pa.array(LZ['nm'][sel].astype(np.float64), pa.float64())}
if S['pol']: cols['ionization_mode'] = pa.array([ZTz.column(S['pol']).cast(pa.string()).to_pylist()[i] for i in sel], pa.string())
if S['ce']: cols['collision_energy_orig'] = pa.array([ZTz.column(S['ce']).cast(pa.string()).to_pylist()[i] for i in sel], pa.string())
KT = pa.table(cols); kfile = 'Z_trainlike.parquet'; pq.write_table(KT, os.path.join(A.out, kfile), compression='zstd')
trSch = pq.read_schema(A.train); six = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
kkeys = set(KT.column('inchikey14').to_pylist()); assert not (kkeys & trainKeys), 'kernel file holds a train key'
R['kernelFile'] = dict(file=kfile, md5=md5(os.path.join(A.out, kfile)), rows=KT.num_rows, keys=len(kkeys), rowsDroppedNoKernelAdduct=int(ZTz.num_rows - KT.num_rows), keysInTrain=len(kkeys & trainKeys),
                       dtypesEqualTrain={c: bool(KT.schema.field(c).type == trSch.field(c).type) for c in six}, dtypes={c: str(KT.schema.field(c).type) for c in KT.schema.names},
                       adductVocab=collections.Counter(KT.column('adduct').to_pylist()).most_common(), adductsOutsideKernelTable=sorted(set(KT.column('adduct').to_pylist()) - set(ns['ADDUCTS'])), hostKeptRows=int(sum(KT.column('hostKeptAdduct').to_pylist())), licence='CC BY 4.0')
R['licence'] = 'CC BY 4.0 (Enveda-180, Zenodo 21346580 v3 2026-07-13); modified subset, see README_LICENSE.txt'
log(f'kernel file: {KT.num_rows:,} rows / {len(kkeys):,} keys, dropped {R["kernelFile"]["rowsDroppedNoKernelAdduct"]:,} rows without a kernel adduct; dtypes equal train {all(R["kernelFile"]["dtypesEqualTrain"].values())}')
open(os.path.join(A.out, 'Z_keys.txt'), 'w').write('\n'.join(Z) + '\n')
open(os.path.join(A.out, 'README_LICENSE.txt'), 'w').write('Enveda-180 (Krettler et al., Enveda Biosciences), Zenodo record 21346580 (v3, 2026-07-13), licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). '
    'MODIFIED: Z_library.parquet / Z_trainlike.parquet hold only the rows whose InChIKey first block is absent from the CASMI26 competition train.parquet; Z_trainlike renames columns to the train.parquet layout (intensities as published). Counts in report.json.\n')
R['outputs'] = {f: dict(md5=md5(os.path.join(A.out, f)), bytes=os.path.getsize(os.path.join(A.out, f))) for f in os.listdir(A.out) if f != 'report.json'}
dup = set(perKey.index[perKey >= 0.99]); R['Z_summary'] = dict(Z_keys=len(Zset), keysWithKernelAdductRow=len(kkeys), keysWithHostKeptAdductRow=len(Zk), keysDuplicateSpectrumInTrainGe099=len(dup & Zset), keysStdCollapseIntoTrain=R['Z_standardise'].get('stdKeyInTrain'),
                      proposal_genuinelyNew=len(Zset) - len(dup & Zset) - (R['Z_standardise'].get('stdKeyInTrain') or 0), note='genuinelyNew = Z keys minus keys with a >= 0.99 duplicate spectrum in train minus keys whose salt-stripped/uncharged form is a train key (overlap of the two not removed twice only if disjoint; see components)')
R['wallSec'] = round(time.time() - T0, 1); json.dump(R, open(os.path.join(A.out, 'report.json'), 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
log(f'done: Z {len(Z):,} keys / {ZTz.num_rows:,} spectra; test molecules >= gate: Z {R["test_vs_Z"]["moleculesGeGate"]} vs control mean {R["test_vs_control_mean"]}; wall {R["wallSec"]} s')
