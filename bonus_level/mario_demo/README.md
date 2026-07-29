# Historical Mario Snapshot

> **Status: frozen and deprecated.** This directory is retained only to
> preserve the original integration history. Do not run, import, test, or
> extend this copy.

The maintained implementation is [`../../bonus_level_mario/`](../../bonus_level_mario/).
It contains the current gesture thresholds, pose adapter, tests, documentation,
and asset set. The root launcher already targets that canonical directory:

```powershell
conda activate vc_sws3026
.\run_mario.ps1 --check
.\run_mario.ps1
```

For development and tests, use:

```powershell
python -m unittest discover -s bonus_level_mario -p "test_*.py" -v
python bonus_level_mario\mario_camera_demo.py --check
```

Why keep this snapshot?

- It records the pre-integration version contributed during the group project.
- Removing the whole directory would erase useful provenance.
- The deprecation boundary prevents two similar implementations from appearing
  equally authoritative.

See the [canonical Mario guide](../../bonus_level_mario/README.md) and the
[project engineering case study](../../PORTFOLIO.md).
