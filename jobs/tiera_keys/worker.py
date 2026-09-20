"""C1 pool worker, job 1: metric keys for one sub-chunk of E3's tier-A pool (same describe() and output schema as
notebooks/candidate_db_tiera_keys/candidate_db_tiera_keys.py, rdkit 2026.03.3), reading the pre-staged pool
(dataset nicholasooo/casmi-c1-tiera-pool, tierAPool.parquet = loadSources() sorted by ik14) instead of train+COCONUT+LOTUS.
Sub-chunk = pool.gather_every(NCHUNKS, offset=CHUNK); CHUNK ≡ 3 (mod 4) with NCHUNKS = 48 partitions E3's chunk 3 of 4.
Idempotent: partial results are checkpointed to <work>/partial.parquet every CKPT_S seconds and reloaded on restart.
Output <work>/out/tierAKeys_c{CHUNK}of{NCHUNKS}.parquet + stats json, uploaded as private dataset nicholasooo/<ds-prefix>-w{CHUNK:02d} (default prefix casmi-c1-tierakeys).
Usage: KAGGLE_API_TOKEN=... python tieraKeysWorker.py --chunk 3 --nchunks 48 --work /data/w3 [--limit N] [--no-upload]
"""
import argparse, json, os, subprocess, sys, time
from multiprocessing import Pool
import polars as pl
import rdkit
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
T0 = time.time()
POOL_DS = 'nicholasooo/casmi-c1-tiera-pool'
SCHEMA = {'ik14': pl.Utf8, 'source': pl.Utf8, 'smiles': pl.Utf8, 'ik14Parent': pl.Utf8, 'ik14Metric': pl.Utf8, 'formula': pl.Utf8, 'monoMass': pl.Float64, 'charge': pl.Int64}
_tools = {}


def log(*a):
    print(f'[{(time.time() - T0) / 60:6.1f} min]', *a, flush=True)


def tools():
    if not _tools:
        _tools.update(taut=rdMolStandardize.TautomerEnumerator(), frag=rdMolStandardize.LargestFragmentChooser(), uncharge=rdMolStandardize.Uncharger())
    return _tools


def describe(smiles):
    """Host-form SMILES + plain key + metric key + formula/mass/charge; None if RDKit cannot parse (verbatim from E3's script)."""
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    try:
        t = tools()
        par = t['uncharge'].uncharge(t['frag'].choose(mol))
        Chem.RemoveStereochemistry(par)
        smi = Chem.MolToSmiles(par)
        par = Chem.MolFromSmiles(smi)
        return dict(smiles=smi, ik14Parent=Chem.MolToInchiKey(par)[:14], ik14Metric=Chem.MolToInchiKey(t['taut'].Canonicalize(par))[:14],
                    formula=rdMolDescriptors.CalcMolFormula(par), monoMass=Descriptors.ExactMolWt(par), charge=Chem.GetFormalCharge(par))
    except (ValueError, RuntimeError, TypeError):
        return None


def work(batch):
    return [(k, describe(s)) for k, s in batch]


def toFrame(df, results):
    rows = [{'ik14': k, 'source': s, **(results.get(k) or {})} for k, s in zip(df['ik14'], df['source']) if k in results]
    return pl.DataFrame(rows, schema=SCHEMA)


def kaggle(*args):
    return subprocess.run(['kaggle', *args], text=True, capture_output=True)


def upload(outDir, dsId, title):
    json.dump({'title': title, 'id': dsId, 'licenses': [{'name': 'CC0-1.0'}]}, open(os.path.join(outDir, 'dataset-metadata.json'), 'w'))
    r = kaggle('datasets', 'create', '-p', outDir, '-q')
    log('create:', (r.stdout + r.stderr).strip()[-300:])
    if r.returncode != 0 or 'already' in (r.stdout + r.stderr).lower() or 'error' in (r.stdout + r.stderr).lower():
        r = kaggle('datasets', 'version', '-p', outDir, '-m', f'rerun {time.strftime("%H:%M UTC", time.gmtime())}', '-q')
        log('version:', (r.stdout + r.stderr).strip()[-300:])
    for i in range(20):  # wait until the files are downloadable (Kaggle processes uploads asynchronously)
        time.sleep(30)
        r = kaggle('datasets', 'files', dsId)
        if 'tierAKeys' in r.stdout:
            log('dataset ready:', dsId); return True
    log('WARNING: dataset not listing files yet:', r.stdout[-300:], r.stderr[-300:]); return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chunk', type=int, required=True); ap.add_argument('--nchunks', type=int, default=48)
    ap.add_argument('--work', default=None); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--procs', type=int, default=2)
    ap.add_argument('--ckpt-s', type=float, default=120); ap.add_argument('--no-upload', action='store_true')
    ap.add_argument('--ds-prefix', default='casmi-c1-tierakeys', help='dataset slug prefix; the worker uploads to nicholasooo/<prefix>-w<CHUNK>')
    a = ap.parse_args()
    work_ = a.work or f'/data/c1w/w{a.chunk:02d}'
    outDir = os.path.join(work_, 'out'); os.makedirs(outDir, exist_ok=True)
    poolPath = os.path.join(work_, 'tierAPool.parquet')
    if not os.path.exists(poolPath):
        r = kaggle('datasets', 'download', POOL_DS, '-f', 'tierAPool.parquet', '-p', work_, '--unzip')
        log('pool download:', (r.stdout + r.stderr).strip()[-200:])
        if os.path.exists(poolPath + '.zip'):
            subprocess.run(['unzip', '-o', '-q', poolPath + '.zip', '-d', work_]); os.remove(poolPath + '.zip')
    pool = pl.read_parquet(poolPath)
    assert pool.height == 729588 and pool['ik14'].is_sorted(), pool.height
    df = pool.gather_every(a.nchunks, offset=a.chunk)
    if a.limit:
        df = df.head(a.limit)
    suffix = f'_c{a.chunk}of{a.nchunks}'
    log(f'rdkit {rdkit.__version__}; chunk {a.chunk}/{a.nchunks}: {df.height} rows')
    ckptPath = os.path.join(work_, 'partial.parquet')
    results = {}
    if os.path.exists(ckptPath):
        part = pl.read_parquet(ckptPath)
        for row in part.iter_rows(named=True):
            k = row.pop('ik14'); row.pop('source')
            results[k] = None if row['ik14Metric'] is None and row['smiles'] is None else row
        log(f'resumed {len(results)} rows from checkpoint')
    items = [(k, s) for k, s in zip(df['ik14'], df['srcSmiles']) if k not in results]
    batches = [items[i:i + 100] for i in range(0, len(items), 100)]
    lastCkpt = time.time()
    with Pool(a.procs) as p:
        for n, res in enumerate(p.imap_unordered(work, batches)):
            for k, d in res:
                results[k] = d
            if n % 20 == 0:
                log(f'{len(results)}/{df.height}')
            if time.time() - lastCkpt > a.ckpt_s:
                toFrame(df, results).write_parquet(ckptPath); lastCkpt = time.time()
    out = toFrame(df, results).with_columns(pl.lit(True).alias('attempted')).sort('monoMass', nulls_last=True)
    assert out.height == df.height, (out.height, df.height)
    out.write_parquet(os.path.join(outDir, f'tierAKeys{suffix}.parquet'), compression='zstd', row_group_size=100000)
    done = out.filter(pl.col('ik14Metric').is_not_null())
    stats = {'rows': out.height, 'attempted': out.height, 'keyed': done.height, 'parseFailed': out.height - done.height,
             'metricKeyChanged': int((done['ik14Metric'] != done['ik14']).sum()), 'parentKeyDiffers': int((done['ik14Parent'] != done['ik14']).sum()),
             'distinctMetricKeys': done['ik14Metric'].n_unique(), 'stoppedByBudget': False, 'rdkit': rdkit.__version__, 'minutes': round((time.time() - T0) / 60, 1),
             'chunk': a.chunk, 'nChunks': a.nchunks, 'worker': 'c1-sandbox', 'procs': a.procs}
    json.dump(stats, open(os.path.join(outDir, f'stats{suffix}.json'), 'w'), indent=1)
    log(json.dumps(stats))
    if not a.no_upload:
        ok = upload(outDir, f'nicholasooo/{a.ds_prefix}-w{a.chunk:02d}', f'{a.ds_prefix}-w{a.chunk:02d}')
        log('DONE' if ok else 'DONE (upload unverified)')


if __name__ == '__main__':
    main()
