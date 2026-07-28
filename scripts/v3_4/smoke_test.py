"""
LEAF v3.4 smoke test: preflight -> invariant tests -> evaluator -> manifest check.

Runs the EVALUATOR as part of the smoke suite (not just the engine internals),
so a broken reporting path fails here rather than in review. Skips cleanly with
a non-fatal message if the preflight reports missing dependencies.

Usage: python scripts/v3_4/smoke_test.py
Exit:  0 all good (or cleanly skipped);  1 a stage failed
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PROD = ROOT / 'data' / 'production'

STAGES = [
    ('invariant tests', [sys.executable, str(HERE / 'test_v34.py')]),
    ('evaluator', [sys.executable, str(HERE / 'evaluate_v34.py')]),
    ('artifact manifest', [sys.executable, str(HERE / 'make_manifest.py')]),
]

EXPECTED_OUTPUTS = [
    ROOT / 'docs' / 'LEAF_V34_RECERTIFICATION.md',
    ROOT / 'docs' / 'LEAF_V34_MANIFEST.md',
    PROD / 'leaf_v34_t1_eval_pairs.csv',
]


def main():
    print('=' * 72)
    print('LEAF v3.4 SMOKE TEST')
    print('=' * 72)

    pre = subprocess.run([sys.executable, str(HERE / 'preflight.py'), '--quiet'])
    if pre.returncode != 0:
        print('\n[SKIP] preflight reported missing dependencies. Run '
              '`python scripts/v3_4/preflight.py` for the actionable list. '
              'Nothing was installed.')
        return 0
    print('[OK] preflight satisfied\n')

    for name, cmd in STAGES:
        print(f'--- {name} ---', flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            print(r.stderr[-3000:])
            print(f'\n[FAIL] stage "{name}" exited {r.returncode}')
            return 1
        tail = [ln for ln in r.stdout.strip().splitlines() if ln.strip()][-3:]
        for ln in tail:
            print('   ' + ln)
        print(f'[OK] {name}\n')

    missing = [p for p in EXPECTED_OUTPUTS if not p.exists()]
    if missing:
        print('[FAIL] expected outputs missing: '
              + ', '.join(p.name for p in missing))
        return 1
    print('[OK] all expected outputs present')
    print('\nSMOKE TEST PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
