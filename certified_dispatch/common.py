"""Shared helpers: repository paths, CSV and atomic file output, provenance of saved runs."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import t as student_t

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
DATA = ROOT / 'data'
CONFIGS = ROOT / 'configs'
NETWORK = DATA / 'manhattan_south.json'
# Arrays of a prepared Manhattan cell, in the order the compiled kernels take them.
INPUT_KEYS = ('support_lo', 'support_hi', 'true_contexts', 'paths', 'lengths', 'free', 'delay')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    """Half-width of the two-sided 95% Student-t interval of the mean; empty string below two values."""
    values = np.asarray(values, float)
    if len(values) < 2:
        return ''
    return float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_source(output):
    """Copy the package sources into `output/source/<package>` and return their SHA-256 digests."""
    target = Path(output) / 'source' / PACKAGE.name
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for path in sorted(PACKAGE.glob('*.py')):
        (target / path.name).write_bytes(path.read_bytes())
        hashes[path.name] = sha(path)
    return hashes


def verify_source(output, manifest):
    """Check the archived source snapshot of a run against the digests in its manifest."""
    source = Path(output) / 'source'
    for filename, digest in manifest['source_sha256'].items():
        candidates = [p for p in (source / PACKAGE.name / filename, source / 'experiments' / filename) if p.exists()]
        if not candidates or sha(candidates[0]) != digest:
            raise ValueError('Archived source does not match the run manifest: ' + filename)


def check_source_unchanged(manifest):
    """Refuse to resume a run with sources that differ from the ones that started it."""
    for filename, digest in manifest['source_sha256'].items():
        path = PACKAGE / filename
        if not path.exists() or sha(path) != digest:
            raise ValueError('Source changed since the run started; resume with the archived source: ' + filename)
