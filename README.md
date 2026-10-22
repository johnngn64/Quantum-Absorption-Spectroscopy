# Molecule identification with light

A Python simulation that distinguishes **acetone from ethanol** using infrared photon counts. It compares coherent light with ideal fixed-photon-number (Fock) light, including missed detections and background noise. It can also estimate an unknown concentration.

This is a completed first simulation version. It does not connect to laboratory equipment or identify arbitrary molecules.

## Run it

Open a terminal in this project folder:

```powershell
python run.py
```

This runs the unknown-concentration comparison with 500 total photons, 90% detector efficiency, and a mean of one background count per frequency. It tests 10,000 samples of each molecule for each light source. It prints results and saves a report; plot windows are off by default.

Each run creates a fresh folder under `runs/`. Open that folder's **report.html** in a browser to see the summary and graphs together. Keep the whole folder when sharing the report because its images are separate files.

The folder also contains:

- `summary.md`: a readable text report.
- `results.json`: settings, numerical results, example counts, confusion matrices, package versions and source/data hashes.
- PNG graphs for the selected experiment.

Earlier results are preserved. Existing PNG files in the project root and the old `validation_results/` folder are earlier outputs; new runs do not update them. Folder timestamps use UTC.

## First-time setup

Tested with Python 3.13. From this folder, create an isolated environment and install the tested package versions:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

When using this environment, substitute `.\.venv\Scripts\python.exe` for `python` in the commands below. Installation needs internet access; simulations use the included reference data locally.

## Useful commands

For the project brief's ideal coherent and Fock relative-entropy comparison, run:

```powershell
python run.py --experiment brief
```

This reports classical relative entropy in nats for coherent (Poisson) and Fock
(binomial) counts at the same incident photon budget, coherent amplitude magnitudes,
entropy versus transmissivity difference and incident photon number, and accumulated
evidence from repeated measurements of the same H0 sample. It uses generic
transmissivities 0.95 and 0.85, perfect detection and zero background.
Use `--transmissivities 0.95 0.85` and `--measurements 20` to change the baseline.
Each repeated measurement uses the full `--photons` mean budget; `--trials` sets
the number of independent evidence experiments. The input phase is arbitrary and
unchanged by pure loss. Positive log evidence favours H0.
The graphs compare both probes' count distributions, entropy sweeps and expected
accumulated evidence; simulated evidence traces are for coherent light.
The HTML, Markdown and JSON reports include both entropy values. Higher entropy
means more expected information per measurement in the H0-to-H1 direction, not
an accuracy percentage. Fock preparation is ideal.
`--experiment both` now includes this baseline alongside the existing nine graphs.
The default `python run.py` still runs the unknown-concentration experiment.

```powershell
# Include all nine graphs from both experiment types
python run.py --experiment both

# More photons, or a different repeatable random seed
python run.py --photons 2000 --seed 7

# Also open graph windows
python run.py --show

# Choose a parent directory; a new child folder is still created
python run.py --output-dir my_results

# See every simulation setting
python run.py simulate --help

# Run validation across photon budgets, efficiency, noise and concentration
python run.py validate

# Run automated checks
python run.py test
```

Validation saves a separate `runs/validation_<timestamp>/` folder containing `validation_report.md`, `validation_results.json`, and three graphs. It checks repeated trials, uncertainty in measured accuracy, no-signal controls, and concentration-grid refinement.

The original `python molecule_identification.py` command still works. It defaults to **both** experiments with graph windows, whereas `python run.py` defaults to **unknown concentration** without windows.

## Reading the results

- **Identity accuracy:** how often the simulated molecule was identified correctly.
- **Concentration MAE:** average absolute concentration error, in micromol/mol. Lower is better; wrong-molecule fits are included.
- **Boundary fits:** estimates at the permitted concentration limits, which may indicate weak information or a restrictive search range.
- **Unidentifiable concentration:** counts provide no concentration information. These estimates are excluded from MAE and stored as `null` in JSON.

With the default seed (42), the unknown-concentration comparison gives approximately **70.58%** identity accuracy for coherent light and **88.68%** for Fock light. These are results of this model, not measured performance of a real device. The same seed, settings, data and software versions reproduce the numerical experiment; timestamps and output paths change on each run.

The classifier chooses only between the two supplied molecules and assumes known path length, detector efficiency and background. Fock-state preparation is ideal. A likelihood tie chooses acetone by convention, not as identifying evidence. More accurate concentration estimates and laboratory validation remain separate future work.

## Project files

| File | Purpose |
| --- | --- |
| `run.py` | Simple command for simulation, validation and tests |
| `molecule_identification.py` | Experiments, command-line settings and graphs |
| `molecule_models.py` | Spectral loading, photon statistics and fitting calculations |
| `simulation_reports.py` | Fresh output folders and HTML/Markdown/JSON reports |
| `validate_molecule_identification.py` | Systematic validation experiments |
| `test_*.py` | Automated checks |
| `requirements.txt` | Tested package versions |
| `data/` | Included NIST spectra and provenance notes |

See [MODEL_NOTES.md](MODEL_NOTES.md) for equations, assumptions, detailed settings and validation methodology, and [data/README.md](data/README.md) for reference-data provenance.
