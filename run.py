"""Simple entry point for the molecule identification simulation."""

import importlib.util
from pathlib import Path
import subprocess
import sys


HELP = """Molecule identification simulation

  python run.py                         Run unknown-concentration comparison; save a report
  python run.py --photons 2000           Change the total incoming photon budget
  python run.py --experiment both        Include the known-concentration experiments
  python run.py --show                   Also display plot windows
  python run.py validate                 Run systematic simulation validation
  python run.py test                     Run all automated checks
  python run.py simulate --help          See all simulation settings
  python run.py validate --help          See all validation settings

Results are saved in separate timestamped folders under runs/.
Setup: python -m pip install -r requirements.txt
"""


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["--help"], ["-h"]):
        print(HELP)
        return 0
    command = arguments.pop(0) if arguments and arguments[0] in ("simulate", "validate", "test") else "simulate"
    missing = [name for name in ("numpy", "scipy", "matplotlib") if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing packages: " + ", ".join(missing), file=sys.stderr)
        print("Install them with: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    if command == "test":
        return subprocess.call([sys.executable, "-m", "unittest", "discover", "-v", *arguments], cwd=Path(__file__).resolve().parent)
    show = "--show" in arguments
    if show:
        arguments.remove("--show")
    if show and "--no-show" in arguments:
        print("Choose either --show or --no-show.", file=sys.stderr)
        return 2
    if not show:
        import matplotlib
        matplotlib.use("Agg")
        arguments.insert(0, "--no-show")
    if command == "validate":
        from validate_molecule_identification import main as execute
    else:
        from molecule_identification import main as execute
        arguments = ["--experiment", "unknown", *arguments]
    execute(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
