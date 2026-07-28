"""
LEAF v3.4 dependency preflight.

Checks the interpreter and every third-party package the v3.4 pipeline imports,
and reports what each one is needed for. It NEVER installs anything and never
downloads data — it only tells you what is missing and what to run.

Exit codes:
  0  everything required is present (data-only gaps are warnings)
  1  a required package is missing or too old

Usage:
  python scripts/v3_4/preflight.py            # check packages + data
  python scripts/v3_4/preflight.py --quiet    # exit code only
"""

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PROD = ROOT / 'data' / 'production'

MIN_PY = (3, 9)

# (module, min_version, required_for, hard_requirement)
PACKAGES = [
    ('numpy', (1, 21), 'array maths in every v3.4 module', True),
    ('pandas', (1, 5), 'all dataframe handling; engine uses only '
                       'pandas-1.5-compatible calls', True),
    ('scipy', (1, 7), 'evaluate_v34.py only (norm/kstest for CRPS, NLL, PIT)', True),
    ('nfl_data_py', (0, 3), 'build_base_v34.py only (nflverse pbp download)', False),
]

# artifacts the later stages need, and what produces them
DATA = [
    (PROD / 'qb_games_base_v34.csv', 'python scripts/v3_4/build_base_v34.py'),
    (ROOT / 'data' / 'raw' / 'player_meta.csv', 'nflverse player metadata refresh'),
    (PROD / 'leaf_v34_ratings.csv', 'python scripts/v3_4/engine_v34.py'),
    (PROD / 'leaf_v34_params.json', 'python scripts/v3_4/engine_v34.py'),
]


def parse_version(v):
    out = []
    for part in str(v).split('.')[:3]:
        digits = ''.join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def main(quiet=False):
    problems, warnings = [], []

    def say(msg):
        if not quiet:
            print(msg)

    say('=' * 72)
    say('LEAF v3.4 PREFLIGHT — checks only; installs nothing, downloads nothing')
    say('=' * 72)

    # ---- interpreter ----
    py = sys.version_info[:3]
    ok = py >= MIN_PY
    say(f'\nPython {".".join(map(str, py))}  '
        f'[{"OK" if ok else "TOO OLD"}] (need >= {".".join(map(str, MIN_PY))})')
    if not ok:
        problems.append(f'Python {".".join(map(str, MIN_PY))}+ required')

    # ---- packages ----
    say('\nPackages:')
    for mod, minv, why, hard in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, '__version__', '?')
            good = ver == '?' or parse_version(ver) >= minv
            tag = 'OK' if good else 'TOO OLD'
            say(f'  [{tag:7s}] {mod:14s} {ver:10s} — {why}')
            if not good:
                msg = (f'{mod} >= {".".join(map(str, minv))} required '
                       f'(found {ver}); install with: pip install "{mod}>='
                       f'{".".join(map(str, minv))}"')
                (problems if hard else warnings).append(msg)
        except ImportError:
            say(f'  [MISSING] {mod:14s} {"—":10s} — {why}')
            msg = f'{mod} not installed; install with: pip install {mod}'
            (problems if hard else warnings).append(msg)

    # ---- data artifacts ----
    say('\nData artifacts:')
    for path, how in DATA:
        if path.exists():
            mb = path.stat().st_size / 1048576
            say(f'  [OK     ] {path.name:34s} {mb:6.2f} MB')
        else:
            say(f'  [MISSING] {path.name:34s} — produce with: {how}')
            warnings.append(f'{path.name} missing; produce with: {how}')

    say('')
    if warnings:
        say('WARNINGS (stage-specific, not fatal for already-built stages):')
        for w in warnings:
            say(f'  - {w}')
    if problems:
        say('\nBLOCKERS:')
        for p in problems:
            say(f'  - {p}')
        say('\nNothing was installed. Run the pip commands above yourself, or ask '
            'for approval before any install.')
        return 1
    say('Preflight OK — required packages satisfied.')
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--quiet', action='store_true')
    sys.exit(main(ap.parse_args().quiet))
