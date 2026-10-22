"""Portable, local run reports and output-directory management."""

from datetime import datetime, timezone
from html import escape
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
CAPTIONS = {
    "brief_coherent_analysis.png": "Coherent (Poisson) versus Fock (binomial) counts and relative entropy at the same incident budget. Expected evidence is shown for both probes; simulated evidence traces are coherent only. Perfect detection and zero background apply regardless of extension settings.",
    "molecule_absorption_spectra.png": "Measured absorption reference spectra; taller peaks mean stronger absorption.",
    "molecule_transmission_spectra.png": "Sample transmission before detector losses; the dashed line marks the single-frequency probe.",
    "molecule_identification_results.png": "Coherent-light recorded counts and decisions for the known-concentration experiment.",
    "molecule_fock_results.png": "Fixed-photon-number recorded counts and decisions for the known-concentration experiment.",
    "molecule_probe_comparison.png": "Known-concentration error versus photon budget. Lines are model predictions; points are simulated trials with 95% Wilson intervals.",
    "molecule_detector_efficiency.png": "Known-concentration error versus detector efficiency, at the selected background level. Points: simulated trials with 95% Wilson intervals.",
    "molecule_background_counts.png": "Known-concentration error versus recorded background noise, at fixed signal settings. Points: simulated trials with 95% Wilson intervals.",
    "molecule_unknown_concentration.png": "Unknown-concentration decisions and estimates. The scatter shows a subset; metrics use all trials.",
    "molecule_concentration_fit.png": "One hidden sample: higher likelihoods fit better. The star selects a fit; the dashed line reveals the truth. These are not confidence probabilities.",
}


def create_run_directory(parent=None, kind="simulation"):
    """Always create a fresh child folder, including when a parent is supplied."""
    parent = ROOT / "runs" if parent is None else Path(parent).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S_%fZ")
    folder = parent / f"{kind}_{timestamp}"
    folder.mkdir(exist_ok=False)
    return folder


def save_figure(figure, filename, output_dir=None):
    target = Path(filename) if output_dir is None else Path(output_dir) / Path(filename).name
    figure.savefig(target, dpi=180)
    return target


def json_value(value):
    """Use strict JSON null for unidentifiable/non-finite numerical results."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def source_hashes():
    paths = [ROOT / name for name in ("molecule_identification.py", "molecule_models.py", "simulation_reports.py", "run.py", "requirements.txt")]
    paths += [ROOT / "data" / name for name in ("acetone_nist.jdx", "ethanol_nist.jdx")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def percent(value):
    if value is None:
        return "Not available"
    return f"{100 * value:.3g}%" if 0 < value < 0.00005 else f"{value:.2%}"


def number(value):
    return "Not identifiable" if value is None else f"{value:.2f}"


def report_sections(settings, experiments):
    sections = []
    if "brief" in experiments:
        baseline = experiments["brief"]
        sections.append({"title": "Project brief — coherent and Fock relative entropy",
                         "text": "Generic H0/H1 absorption hypotheses; perfect detection and zero background. "
                         "D(P0 || P1) uses natural logarithms (nats). Independent measurements give total entropy mD. "
                         "Evidence is accumulated for one unchanged H0 sample; positive values favour H0. "
                         "Coherent counts are Poisson; Fock counts are binomial, at the same incident budget (mean for coherent, exact for Fock). "
                         "Higher relative entropy means more expected information per measurement in this direction, not an accuracy percentage. "
                         "Amplitude magnitudes apply only to coherent light; phase is unchanged.",
                         "headers": ["Quantity", "Value"],
                         "rows": [["Transmissivities (H0, H1)", str(baseline["transmissivities"])],
                                  ["Mean detected counts (H0, H1)", str(baseline["mean_counts"])],
                                  ["Coherent relative entropy (nats)", f'{baseline["relative_entropy_nats"]:.6g}'],
                                  ["Fock relative entropy (nats)", f'{baseline["fock_relative_entropy_nats"]:.6g}'],
                                  ["Input amplitude magnitude", f'{baseline["alpha_in_magnitude"]:.6g}'],
                                  ["Output amplitude magnitudes (H0, H1)", str(baseline["alpha_out_magnitudes"])]]})
    for key, title in (("known", "Known concentration — single frequency"),
                       ("unknown", "Unknown molecule and concentration — multiple frequencies")):
        if key not in experiments:
            continue
        experiment = experiments[key]
        if key == "known":
            text = (f"Concentration {settings['concentration']:g} µmol/mol is supplied to the classifier. "
                    f"Probe frequency: {settings['wavenumber']:g} cm⁻¹. Photon budget: {settings['photons']} per measurement.")
            headers = ["Probe", "Simulated accuracy", "Simulated error", "Theoretical error"]
            rows = [[label, percent(experiment[mode]["accuracy"]), percent(1 - experiment[mode]["accuracy"]),
                     percent(experiment[mode]["theoretical_error"])]
                    for mode, label in (("coherent", "Coherent light"), ("fock", "Fock state"))]
        else:
            text = (f"The classifier fits both identity and concentration. Frequencies: {experiment['frequencies']} cm⁻¹; "
                    f"total photon budget {settings['photons']} split as {experiment['photon_allocation']}. "
                    f"True concentrations are drawn uniformly over {settings['concentration_range']} µmol/mol.")
            headers = ["Probe", "Identity accuracy", "Concentration MAE (µmol/mol)", "Boundary fits", "Unidentifiable concentration"]
            rows = [[label, percent(experiment[mode]["accuracy"]), number(experiment[mode]["concentration_mae"]),
                     percent(experiment[mode]["boundary_rate"]), percent(experiment[mode]["unidentifiable_rate"])]
                    for mode, label in (("coherent", "Coherent light"), ("fock", "Fock state"))]
        sections.append({"title": title, "text": text, "headers": headers, "rows": rows})
        if key == "unknown":
            examples = []
            for mode, label in (("coherent", "Coherent light"), ("fock", "Fock state")):
                example = experiment[mode]["example"]
                flags = []
                if example.get("tied"):
                    flags.append("Molecule tie; acetone chosen by convention")
                if example.get("boundary"):
                    flags.append("Concentration at range boundary")
                examples.append([label, str(example["counts"]), example["predicted_molecule"],
                                 number(example["estimated_concentration"]), example["true_molecule"],
                                 number(example["true_concentration"]), "; ".join(flags) or "None"])
            sections.append({"title": "One hidden-sample example", "text": "Truth is revealed after fitting. An example can be worse than average.",
                             "headers": ["Probe", "Recorded counts", "Predicted molecule", "Estimated concentration", "True molecule", "True concentration", "Fit flags"],
                             "rows": examples})
    return sections


def save_run_report(args, experiments):
    folder = Path(args.output_dir)
    settings = json_value(vars(args))
    experiments = json_value(experiments)
    graphs = [name for name in CAPTIONS if (folder / name).is_file()]
    payload = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
               "settings": settings, "experiments": experiments, "plots": graphs,
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
               "sha256": source_hashes()}
    (folder / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    sections = report_sections(settings, experiments)
    conditions = (f"Seed {args.seed} · {args.trials:,} trials per molecule · Detector efficiency {args.detector_efficiency:.0%} · "
                  f"Background {args.background_counts:g} recorded counts per frequency · Path length {args.path_length:g} m")
    limits = [
        "These are simulated acetone-versus-ethanol samples, not independent laboratory measurements.",
        "Coherent photon budgets are means; Fock budgets are exact incident numbers. Source preparation is ideal.",
        "Detector efficiency, background and path length are known to the classifier. Concentration is hidden only in the unknown experiment.",
        "Concentration MAE includes wrong-molecule fits and excludes unidentifiable estimates. Boundary fits are constrained by the search range.",
        "Finite-trial accuracy can fluctuate. Zero observed mistakes do not establish zero theoretical error.",
        "Known- and unknown-concentration experiments differ in frequencies, sample concentrations and total background exposure; compare probes within each experiment.",
    ]
    if "brief" in experiments:
        limits.insert(0, "The brief baseline uses generic transmissivities, not molecular reference spectra. It always uses perfect detection and zero background; header detector settings apply to molecule experiments when present.")
    if set(experiments) == {"brief"}:
        conditions = f"Seed {args.seed} · {args.trials:,} H0 evidence experiments · Perfect detection · Zero background"
        limits = ["The baseline uses generic weak/strong absorption hypotheses, not identified molecules.",
                  "Relative entropy is directional classical information after photon counting, not quantum relative entropy.",
                  "Each repeated measurement uses the full incident mean photon budget; m measurements use m times that budget."]
    markdown = ["# Molecule identification — simulation report", "", conditions, ""]
    body = [f"<header><p class='eyebrow'>ASP QUANTUM · SIMULATION</p><h1>Molecule identification</h1><p>{escape(conditions)}</p></header>"]
    for section in sections:
        markdown += [f"## {section['title']}", "", section["text"], "",
                     "| " + " | ".join(section["headers"]) + " |",
                     "| " + " | ".join("---" for _ in section["headers"]) + " |"]
        markdown += ["| " + " | ".join(row) + " |" for row in section["rows"]] + [""]
        table = "<tr>" + "".join(f"<th>{escape(h)}</th>" for h in section["headers"]) + "</tr>"
        table += "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in section["rows"])
        body.append(f"<section><h2>{escape(section['title'])}</h2><p>{escape(section['text'])}</p><div class='table'><table>{table}</table></div></section>")
    markdown += ["## How to interpret this run", ""] + ["- " + note for note in limits] + ["", "## Graphs", ""]
    body.append("<section><h2>How to interpret this run</h2><ul>" + "".join(f"<li>{escape(note)}</li>" for note in limits) + "</ul></section>")
    for name in graphs:
        markdown += [f"![{CAPTIONS[name]}]({name})", "", CAPTIONS[name], ""]
        body.append(f"<figure><img src='{escape(name, quote=True)}' alt='{escape(CAPTIONS[name], quote=True)}'><figcaption>{escape(CAPTIONS[name])}</figcaption></figure>")
    markdown += ["Full settings, example counts, confusion matrices, versions and file hashes are saved in [results.json](results.json).", ""]
    body.append("<footer><a href='results.json'>Full results and settings (JSON)</a> · <a href='summary.md'>Text report</a></footer>")
    css = """body{margin:0;background:#edf2f6;color:#192936;font:16px/1.6 Segoe UI,Arial,sans-serif}main{max-width:1180px;margin:auto;padding:32px 24px}header,section,figure{background:white;padding:26px;margin:0 0 22px;border-radius:10px}header{border-top:6px solid #247c92}h1{font-size:36px;margin:4px 0}h2{font-size:22px;margin:0 0 10px}.eyebrow{font-size:12px;letter-spacing:2px;color:#247c92}p{margin:8px 0 18px}.table{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:12px;border-bottom:1px solid #dbe4eb}th{background:#f4f7f9}img{width:100%;height:auto}figcaption,footer{color:#526575}a{color:#146b85}li{margin-bottom:7px}footer{padding:12px 0}@media print{body{background:white}main{padding:0}section,figure{break-inside:avoid;border:1px solid #ddd}}"""
    html = "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
    html += f"<title>Molecule identification — simulation report</title><style>{css}</style></head><body><main>" + "".join(body) + "</main></body></html>"
    (folder / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    (folder / "report.html").write_text(html, encoding="utf-8")
    print(f"\nRun saved to: {folder}\nOpen report.html to see the summary and all graphs together.")
    return payload
