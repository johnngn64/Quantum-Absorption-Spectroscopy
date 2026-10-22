"""Identify molecules from their interaction with light using quantum theory.

Compare coherent and fixed-photon-number light for identifying simulated molecules.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from simulation_reports import create_run_directory, save_figure, save_run_report

# Re-export model functions for existing scripts and notebooks.
from molecule_models import (
    DATA_DIR,
    ACETONE_SPECTRUM,
    ETHANOL_SPECTRUM,
    UNCERTIFIED_REGIONS,
    NUMBER,
    load_spectrum,
    absorption_to_transmission,
    transmission_at_wavenumber,
    FockWithBackground,
    count_models,
    detection_probability,
    classify_counts,
    simulate_identification,
    theoretical_error,
    detector_error_curve,
    background_error_curve,
    allocate_photons,
    fit_unknown_sample,
    draw_unknown_counts,
    poisson_relative_entropy,
    binomial_relative_entropy,
    wilson_interval,
)


OUTPUT_PLOT = Path(__file__).resolve().parent / "molecule_absorption_spectra.png"
TRANSMISSION_PLOT = Path(__file__).resolve().parent / "molecule_transmission_spectra.png"
IDENTIFICATION_PLOT = Path(__file__).resolve().parent / "molecule_identification_results.png"
FOCK_PLOT = Path(__file__).resolve().parent / "molecule_fock_results.png"
COMPARISON_PLOT = Path(__file__).resolve().parent / "molecule_probe_comparison.png"
DETECTOR_PLOT = Path(__file__).resolve().parent / "molecule_detector_efficiency.png"
BACKGROUND_PLOT = Path(__file__).resolve().parent / "molecule_background_counts.png"
UNKNOWN_PLOT = Path(__file__).resolve().parent / "molecule_unknown_concentration.png"
FIT_PLOT = Path(__file__).resolve().parent / "molecule_concentration_fit.png"

# Illustrative model settings, applied equally to both candidate molecules.
DEFAULT_CONCENTRATION = 100.0  # micromol/mol
DEFAULT_PATH_LENGTH = 1.0      # metres


def plot_identification(names, expected_counts, result, probe, photons, fock_photons=None,
                        detector_efficiency=1.0, background_counts=0.0, output_dir=None):
    """Show count-distribution overlap and the repeated-trial decisions."""
    fig, (count_ax, decision_ax) = plt.subplots(1, 2, figsize=(12, 5.5))
    models = count_models(expected_counts, fock_photons, background_counts)
    low = min(int(model.ppf(1e-6)) for model in models)
    high = max(int(model.ppf(1 - 1e-6)) for model in models)
    low, high = min(low, result["observed"]), max(high, result["observed"])
    # Cap the number of plotted points for large photon budgets.
    counts = np.unique(np.linspace(low, high, min(high - low + 1, 4000), dtype=int))
    for name, mean, model in zip(names, expected_counts, models):
        count_ax.plot(counts, model.pmf(counts), marker="." if fock_photons is not None else None,
                      label=f"{name}: mean {model.mean():.1f}")
    count_ax.axvline(result["observed"], color="black", linestyle="--",
                     label=f"Example count: {result['observed']}")
    count_ax.set(xlabel="Recorded count (signal + background)", ylabel="Probability per count",
                 title=f"Count distributions at {probe:g} cm⁻¹")
    count_ax.grid(True, alpha=0.25)
    count_ax.legend(fontsize=9)

    confusion = result["confusion"]
    fractions = confusion / confusion.sum(axis=1, keepdims=True)
    decision_ax.imshow(fractions, vmin=0, vmax=1, cmap="Blues")
    for actual in range(2):
        for predicted in range(2):
            decision_ax.text(predicted, actual,
                             f"{confusion[actual, predicted]:,}\n{fractions[actual, predicted]:.1%}",
                             ha="center", va="center",
                             color="white" if fractions[actual, predicted] > 0.5 else "black")
    decision_ax.set_xticks([0, 1], names)
    decision_ax.set_yticks([0, 1], names)
    decision_ax.set(xlabel="Predicted molecule", ylabel="True molecule",
                    title=f"Simulated accuracy: {result['accuracy']:.2%}")
    title = (f"Coherent-light baseline • {photons:g} mean incoming photons per measurement"
             if fock_photons is None else
             f"Fock-state probe • Exactly {fock_photons} incoming photons per measurement")
    fig.suptitle(title)
    fig.text(0.5, 0.02,
             f"Equal priors • Detector efficiency: {detector_efficiency:.0%} • Mean background: {background_counts:g} counts/measurement",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, IDENTIFICATION_PLOT if fock_photons is None else FOCK_PLOT, output_dir)
    return fig


def plot_simulated_errors(ax, x, coherent_result, fock_result):
    """Mark finite-trial error rates with 95% Wilson intervals over all trials."""
    for result, colour, marker, label in ((coherent_result, "C0", "o", "Coherent simulation"),
                                          (fock_result, "C1", "s", "Fock simulation")):
        confusion = result["confusion"]
        total = int(confusion.sum())
        errors = total - int(np.trace(confusion))
        low, high = wilson_interval(errors, total)
        error = errors / total
        ax.errorbar([x], [100 * error], yerr=[[100 * (error - low)], [100 * (high - error)]],
                    fmt=marker, color=colour, markeredgecolor="black", markersize=7, capsize=4,
                    zorder=4, label=f"{label} (95% Wilson interval)")


def plot_probe_comparison(transmissions, photons, sweep_max, coherent_result, fock_result,
                          probe, concentration, path_length, detector_efficiency=1.0,
                          background_counts=0.0, output_dir=None):
    """Compare theoretical errors at matched integer mean incident budgets."""
    budgets = np.unique(np.append(np.linspace(0, max(sweep_max, photons), 61, dtype=int), photons))
    coherent_errors, fock_errors = [], []
    probabilities = detection_probability(transmissions, detector_efficiency)
    for budget in budgets:
        means = budget * probabilities
        coherent_errors.append(theoretical_error(means, background_counts=background_counts))
        fock_errors.append(theoretical_error(means, int(budget), background_counts))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(budgets, 100 * np.asarray(coherent_errors), label="Coherent light: theoretical error")
    ax.plot(budgets, 100 * np.asarray(fock_errors), label="Fock state: theoretical error")
    plot_simulated_errors(ax, photons, coherent_result, fock_result)
    ax.set(xlabel="Incoming photon budget (coherent mean / exact Fock number)",
           ylabel="Identification error (%)", ylim=(-1, 52),
           title=f"Photon-counting comparison at {probe:g} cm⁻¹\n"
                 f"Concentration = {concentration:g} µmol/mol; path length = {path_length:g} m")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.02,
             f"Detector efficiency: {detector_efficiency:.0%} • Background: {background_counts:g} counts/measurement • Lines: theory; points: trials with 95% Wilson intervals",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, COMPARISON_PLOT, output_dir)
    return fig


def plot_detector_comparison(transmissions, photons, efficiency, coherent_result, fock_result,
                             probe, concentration, path_length, background_counts=0.0, output_dir=None):
    """Show the effect of independently missed photons on both probes."""
    efficiencies = np.unique(np.append(np.linspace(0, 1, 51), efficiency))
    coherent_errors, fock_errors = detector_error_curve(transmissions, photons, efficiencies, background_counts)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(100 * efficiencies, 100 * coherent_errors, label="Coherent light: theoretical error")
    ax.plot(100 * efficiencies, 100 * fock_errors, label="Fock state: theoretical error")
    ax.axvline(100 * efficiency, color="gray", linestyle="--", linewidth=1,
               label=f"Selected efficiency: {efficiency:.0%}")
    plot_simulated_errors(ax, 100 * efficiency, coherent_result, fock_result)
    ax.set(xlabel="Detector efficiency (%) — fraction of arriving photons counted",
           ylabel="Identification error (%)", xlim=(-1, 101), ylim=(-1, 52),
           title=f"Missed detections at {probe:g} cm⁻¹ • {photons} incoming photon budget\n"
                 f"Concentration = {concentration:g} µmol/mol; path length = {path_length:g} m")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.02,
             f"Same known efficiency for both probes • Mean background: {background_counts:g} counts/measurement",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, DETECTOR_PLOT, output_dir)
    return fig


def plot_background_comparison(expected_counts, photons, background, sweep_max,
                               coherent_result, fock_result, probe, efficiency,
                               concentration, path_length, output_dir=None):
    levels = np.unique(np.append(np.linspace(0, max(sweep_max, background), 41), background))
    coherent_errors, fock_errors = background_error_curve(expected_counts, photons, levels)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(levels, 100 * coherent_errors, label="Coherent light: theoretical error")
    ax.plot(levels, 100 * fock_errors, label="Fock state: theoretical error")
    ax.axvline(background, color="gray", linestyle="--", linewidth=1,
               label=f"Selected mean background: {background:g}")
    plot_simulated_errors(ax, background, coherent_result, fock_result)
    ax.set(xlabel="Mean recorded background counts per measurement",
           ylabel="Identification error (%)", ylim=(-1, 52),
           title=f"Background noise at {probe:g} cm⁻¹ • {photons} incoming photon budget\n"
                 f"Detector efficiency: {efficiency:.0%}; concentration: {concentration:g} µmol/mol; path: {path_length:g} m")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.02,
             "Independent Poisson background • Same known noise for both probes • Lines: theory; points: trials with 95% Wilson intervals",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, BACKGROUND_PLOT, output_dir)
    return fig


def run_unknown_experiment(args):
    """Run paired unknown-concentration trials and save evaluation/example plots."""
    probes = np.asarray(args.probe_wavenumbers, dtype=float)
    grid = np.linspace(*args.concentration_range, args.concentration_grid)
    budgets = allocate_photons(args.photons, len(probes))
    names, coefficients = [], []
    for path in (ACETONE_SPECTRUM, ETHANOL_SPECTRUM):
        name, x, a = load_spectrum(path)
        for probe in probes:
            transmission_at_wavenumber(x, a, probe, 0, 1)  # Validate range and excluded regions.
        names.append(name)
        coefficients.append(np.maximum(np.interp(probes, x, a), 0))
    coefficients = np.asarray(coefficients)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 10]))
    labels = np.repeat(np.arange(2), args.trials)
    true_concentrations = rng.uniform(grid[0], grid[-1], size=len(labels))
    example_label = int(rng.integers(2)) if args.true_molecule is None else names.index(args.true_molecule.title())
    example_concentration = (rng.uniform(grid[0], grid[-1]) if args.true_concentration is None
                             else args.true_concentration)
    # Example comes first; paired test samples are identical for both probe types.
    all_labels = np.insert(labels, 0, example_label)
    all_concentrations = np.insert(true_concentrations, 0, example_concentration)

    print("\nUnknown-concentration experiment (separate from the known-concentration baseline)")
    print(f"Total incident budget: {args.photons}; frequencies: {probes.tolist()} cm^-1; split: {budgets.tolist()}")
    print(f"Concentration fit: {grid[0]:g} to {grid[-1]:g} micromol/mol, {len(grid)} grid points")
    print("Trials use continuous uniform concentrations in this range, shared between probe types.")
    print(f"Background mean: {args.background_counts:g} per frequency measurement; "
          f"{len(probes) * args.background_counts:g} total per sample.")
    print("The classifier knows path length, detector efficiency and background; concentration is fitted.")
    evaluation_fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fit_fig, fit_axes = plt.subplots(1, 2, figsize=(12, 5))

    summary = {"frequencies": probes, "photon_allocation": budgets, "molecule_order": names}
    for column, fock in enumerate((False, True)):
        label = "Fock state" if fock else "Coherent light"
        observed = draw_unknown_counts(
            coefficients, all_labels, all_concentrations, budgets, args.path_length,
            args.detector_efficiency, args.background_counts,
            np.random.SeedSequence([args.seed, 11 + column]), fock,
        )
        fitted = fit_unknown_sample(
            observed[1:], coefficients, grid, budgets, args.path_length,
            args.detector_efficiency, args.background_counts, fock,
        )
        example = fit_unknown_sample(
            observed[:1], coefficients, grid, budgets, args.path_length,
            args.detector_efficiency, args.background_counts, fock, return_likelihoods=True,
        )
        confusion = np.zeros((2, 2), dtype=int)
        np.add.at(confusion, (labels, fitted["predicted"]), 1)
        accuracy = np.mean(fitted["predicted"] == labels)
        valid = np.isfinite(fitted["concentration"])
        mae = np.mean(np.abs(fitted["concentration"][valid] - true_concentrations[valid])) if valid.any() else np.nan
        summary["fock" if fock else "coherent"] = {
            "accuracy": accuracy, "confusion": confusion, "concentration_mae": mae,
            "boundary_rate": np.mean(fitted["boundary"]),
            "unidentifiable_rate": np.mean(~valid), "tie_rate": np.mean(fitted["tied"]),
            "example": {"counts": observed[0], "predicted_molecule": names[example["predicted"][0]],
                        "estimated_concentration": example["concentration"][0],
                        "true_molecule": names[example_label], "true_concentration": example_concentration,
                        "boundary": example["boundary"][0], "tied": example["tied"][0]},
        }
        estimate = example["concentration"][0]
        estimate_text = f"{estimate:.2f} micromol/mol" if np.isfinite(estimate) else "not identifiable"
        print(f"\n{label}: example recorded counts {observed[0].tolist()}")
        print(f"  Predicted {names[example['predicted'][0]]}; fitted concentration {estimate_text}")
        print(f"  Revealed truth: {names[example_label]}, {example_concentration:.2f} micromol/mol")
        if example["tied"][0]:
            print("  Molecule likelihoods are tied; Acetone is the tie-break choice, not identifying evidence.")
        if example["boundary"][0]:
            print("  Example estimate hits the fit-range boundary.")
        print(f"  Accuracy across {len(labels):,} trials: {accuracy:.2%}")
        mae_text = f"{mae:.2f} micromol/mol" if valid.any() else "not identifiable"
        print(f"  Concentration mean absolute error (all identifiable fits, including wrong molecules): {mae_text}")
        print(f"  Fits at a range boundary: {np.mean(fitted['boundary']):.2%}; "
              f"molecule ties: {np.mean(fitted['tied']):.2%}; unidentifiable concentrations: {np.mean(~valid):.2%}")
        print(f"  True/predicted counts (rows/columns: Acetone, Ethanol): {confusion.tolist()}")

        ax = axes[0, column]
        fractions = confusion / args.trials
        ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{confusion[i,j]:,}\n{fractions[i,j]:.1%}", ha="center", va="center",
                        color="white" if fractions[i,j] > 0.5 else "black")
        ax.set_xticks([0, 1], names)
        ax.set_yticks([0, 1], names)
        ax.set(xlabel="Predicted molecule", ylabel="True molecule", title=f"{label}: {accuracy:.2%} correct")
        ax = axes[1, column]
        for molecule in range(2):
            selection = np.flatnonzero((labels == molecule) & valid)[::max(1, args.trials // 500)]
            ax.scatter(true_concentrations[selection], fitted["concentration"][selection],
                       s=9, alpha=0.35, label=f"True {names[molecule]}")
        ax.plot([grid[0], grid[-1]], [grid[0], grid[-1]], "k--", linewidth=1, label="Perfect estimate")
        ax.set(xlabel="True concentration (µmol/mol)", ylabel="Estimated concentration (µmol/mol)",
               title=f"Mean absolute concentration error: {mae:.1f} µmol/mol" if valid.any()
               else "Concentration is not identifiable")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        ax = fit_axes[column]
        likelihoods = example["log_likelihoods"][0]
        relative = likelihoods - np.max(likelihoods)
        for molecule, name in enumerate(names):
            ax.plot(grid, relative[molecule], label=name)
        ax.axvline(example_concentration, color="gray", linestyle="--", label="True concentration (revealed)")
        if np.isfinite(estimate):
            ax.plot(estimate, 0, "k*", markersize=10, label="Selected fit")
        ax.set(xlabel="Candidate concentration (µmol/mol)", ylabel="Log likelihood relative to best fit",
               title=f"{label}: counts {observed[0].tolist()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    conditions = (f"{args.photons} total photons split {budgets.tolist()} at {probes.tolist()} cm⁻¹\n"
                  f"Efficiency {args.detector_efficiency:.0%}; background {args.background_counts:g} per frequency; path {args.path_length:g} m")
    evaluation_fig.suptitle("Unknown molecule and concentration\n" + conditions)
    evaluation_fig.text(0.5, 0.01, "Paired simulated samples • Uniform concentration range • Joint maximum-likelihood grid fit",
                        ha="center", fontsize=9, color="dimgray")
    evaluation_fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    save_figure(evaluation_fig, UNKNOWN_PLOT, args.output_dir)
    fit_fig.suptitle(f"One hidden sample: {names[example_label]}, {example_concentration:.2f} µmol/mol (truth revealed after fitting)")
    fit_fig.text(0.5, 0.01, "Higher curves fit better • Curves are likelihoods, not posterior probabilities or confidence intervals",
                 ha="center", fontsize=9, color="dimgray")
    fit_fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_figure(fit_fig, FIT_PLOT, args.output_dir)
    print(f"Saved unknown-concentration results: {args.output_dir / UNKNOWN_PLOT.name}")
    print(f"Saved example fit: {args.output_dir / FIT_PLOT.name}")
    return [evaluation_fig, fit_fig], summary


def run_brief_experiment(args):
    """Compare ideal coherent and Fock probes under the same pure loss."""
    eta0, eta1 = args.transmissivities
    means = args.photons * np.array([eta0, eta1])
    divergence = float(poisson_relative_entropy(*means))
    fock_divergence = float(binomial_relative_entropy(args.photons, eta0, eta1))
    alpha_in = np.sqrt(args.photons)
    alpha_out = np.sqrt([eta0, eta1]) * alpha_in
    steps = np.arange(1, args.measurements + 1)
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 30]))
    # Accumulate evidence for the SAME hypothesis across measurements, without
    # allocating trials x measurements arrays. Pure loss preserves input phase.
    evidence = np.zeros(args.trials)
    average, example = [], []
    for _ in steps:
        counts = rng.poisson(means[0], size=args.trials)
        increment = np.zeros(args.trials) if means[0] == 0 else (
            counts * np.log(eta0 / eta1) + means[1] - means[0])
        evidence += increment
        average.append(float(evidence.mean()))
        example.append(float(evidence[0]))
    eta1_sweep = np.linspace(eta0, min(eta1, eta0 * 0.5), 101)
    gap = eta0 - eta1_sweep
    gap_entropy = poisson_relative_entropy(args.photons * eta0, args.photons * eta1_sweep)
    budgets = np.unique(np.append(np.linspace(0, max(args.sweep_max, args.photons), 101, dtype=int), args.photons))
    budget_entropy = poisson_relative_entropy(budgets * eta0, budgets * eta1)
    fock_gap_entropy = binomial_relative_entropy(args.photons, eta0, eta1_sweep)
    fock_budget_entropy = binomial_relative_entropy(budgets, eta0, eta1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    models = count_models(means)
    low = min(int(m.ppf(1e-6)) for m in models)
    high = max(int(m.ppf(1 - 1e-6)) for m in models)
    counts = np.unique(np.linspace(low, high, min(high - low + 1, 4000), dtype=int))
    for label, model in zip(("H0: weak absorption", "H1: stronger absorption"), models):
        axes[0, 0].plot(counts, model.pmf(counts), label=f"Coherent {label}")
    for label, model in zip(("H0", "H1"), count_models(means, args.photons)):
        fock_counts = np.arange(int(model.ppf(1e-6)), int(model.ppf(1 - 1e-6)) + 1)
        axes[0, 0].plot(fock_counts, model.pmf(fock_counts), "--", label=f"Fock {label}")
    axes[0, 0].set(xlabel="Detected photons", ylabel="Probability per count", title="Same incident photon budget; ideal detection")
    axes[0, 0].legend()
    axes[0, 1].plot(gap, gap_entropy, label="Coherent (Poisson)")
    axes[0, 1].plot(gap, fock_gap_entropy, label="Fock (binomial)")
    axes[0, 1].legend()
    axes[0, 1].set(xlabel="Transmission difference (eta0 - eta1)", ylabel="Relative entropy (nats)", title=f"Fixed eta0 = {eta0:g}; fixed mean photons = {args.photons}")
    axes[1, 0].plot(budgets, budget_entropy, label="Coherent (Poisson)")
    axes[1, 0].plot(budgets, fock_budget_entropy, label="Fock (binomial)")
    axes[1, 0].legend()
    axes[1, 0].set(xlabel="Incident photons (coherent mean / exact Fock number)", ylabel="Relative entropy (nats)", title="Fixed pair of transmissivities")
    axes[1, 1].plot(steps, example, alpha=0.6, label="Coherent: one H0 experiment")
    axes[1, 1].plot(steps, average, label=f"Coherent: mean of {args.trials} experiments")
    axes[1, 1].plot(steps, steps * divergence, "--", label="Coherent: expected mD")
    axes[1, 1].plot(steps, steps * fock_divergence, "--", label="Fock: expected mD")
    axes[1, 1].set(xlabel="Independent measurements of the same sample", ylabel="Cumulative log evidence (nats)", title="Positive evidence favours H0")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Pure loss: coherent D = {divergence:.4f} nats; Fock D = {fock_divergence:.4f} nats")
    fig.text(0.5, 0.015, f"Perfect detection; zero background. |alpha_in| = {alpha_in:.3g}; "
             f"|alpha_out| = {alpha_out[0]:.3g} (H0), {alpha_out[1]:.3g} (H1). Coherent amplitudes only.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    save_figure(fig, "brief_coherent_analysis.png", args.output_dir)
    summary = {"transmissivities": [eta0, eta1], "mean_counts": means,
               "detector_efficiency": 1.0, "background_counts": 0.0,
               "alpha_in_magnitude": alpha_in, "alpha_out_magnitudes": alpha_out,
               "relative_entropy_nats": divergence, "measurements": steps,
               "fock_relative_entropy_nats": fock_divergence,
               "fock_total_relative_entropy_nats": steps * fock_divergence,
               "fock_transmission_sweep_entropy_nats": fock_gap_entropy,
               "fock_photon_sweep_entropy_nats": fock_budget_entropy,
               "total_relative_entropy_nats": steps * divergence,
               "mean_log_evidence_h0": average, "example_log_evidence_h0": example,
               "transmission_difference": gap, "transmission_sweep_entropy_nats": gap_entropy,
               "photon_budgets": budgets, "photon_sweep_entropy_nats": budget_entropy}
    print(f"\nIdeal coherent baseline: eta0={eta0:g}, eta1={eta1:g}; D(P0 || P1)={divergence:.6g} nats")
    print(f"Ideal Fock comparison: N={args.photons}; D(P0 || P1)={fock_divergence:.6g} nats (same transmissions and incident budget)")
    print(f"Amplitude magnitudes: input={alpha_in:.6g}; output={alpha_out}; phase unchanged")
    return fig, summary


def main(argv=None):
    """Plot spectra and identify simulated samples by photon counting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("brief", "known", "unknown", "both"), default="both",
                        help="Run brief baseline, known/unknown concentration, or both plus baseline (default: both).")
    parser.add_argument("--transmissivities", nargs=2, type=float, default=[0.95, 0.85],
                        metavar=("ETA0", "ETA1"), help="Brief baseline: 0 < ETA1 <= ETA0 <= 1 (default: 0.95 0.85).")
    parser.add_argument("--measurements", type=int, default=20,
                        help="Repeated measurements per H0 evidence experiment (default: 20).")
    parser.add_argument("--probe-wavenumbers", nargs="+", type=float, default=[1050.0, 1200.0],
                        help="At least two distinct frequencies for unknown concentration (default: 1050 1200).")
    parser.add_argument("--concentration-range", nargs=2, type=float, default=[10.0, 200.0],
                        metavar=("LOW", "HIGH"), help="Unknown-concentration range in micromol/mol (default: 10 200).")
    parser.add_argument("--concentration-grid", type=int, default=101,
                        help="Number of fitted concentration grid points (default: 101).")
    parser.add_argument("--true-concentration", type=float,
                        help="Hidden concentration for the unknown example only (default: random in range).")
    parser.add_argument(
        "--no-show", action="store_true", help="Save the plots without opening windows."
    )
    parser.add_argument(
        "--concentration", type=float, default=DEFAULT_CONCENTRATION,
        help="Concentration for both molecules in micromol/mol (default: 100).",
    )
    parser.add_argument(
        "--path-length", type=float, default=DEFAULT_PATH_LENGTH,
        help="Optical path length in metres (default: 1).",
    )
    parser.add_argument("--wavenumber", type=float, default=1050.0,
                        help="Probe wavenumber in cm^-1 (default: 1050).")
    parser.add_argument("--photons", type=float, default=500.0,
                        help="Integer photon budget: coherent mean / exact Fock number (default: 500).")
    parser.add_argument("--sweep-max", type=int, default=1000,
                        help="Maximum comparison budget, expanded to include --photons (default: 1000).")
    parser.add_argument("--detector-efficiency", type=float, default=0.9,
                        help="Fraction of arriving photons counted, from 0 to 1 (default: 0.9).")
    parser.add_argument("--background-counts", type=float, default=1.0,
                        help="Mean recorded Poisson background counts per measurement (default: 1).")
    parser.add_argument("--background-max", type=float, default=100.0,
                        help="Background sweep maximum, expanded to include selected level (default: 100).")
    parser.add_argument("--trials", type=int, default=10000,
                        help="Independent test measurements per molecule (default: 10000).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for repeatable simulations (default: 42).")
    parser.add_argument("--true-molecule", choices=("acetone", "ethanol"),
                        help="True sample for the single example (default: randomly chosen).")
    parser.add_argument("--output-dir", type=Path, help="Parent folder for a fresh run folder (default: runs).")
    args = parser.parse_args(argv)
    eta0, eta1 = args.transmissivities
    if not np.all(np.isfinite([eta0, eta1])) or not 0 < eta1 <= eta0 <= 1:
        parser.error("Brief transmissivities require 0 < ETA1 <= ETA0 <= 1")
    if args.measurements < 1:
        parser.error("--measurements must be positive")
    if args.experiment == "brief":
        args.detector_efficiency, args.background_counts = 1.0, 0.0
    try:
        detection_probability([1, 1], args.detector_efficiency)
        count_models([0, 0], background_counts=args.background_counts)
    except ValueError as error:
        parser.error(str(error))
    for name, value in (("--concentration", args.concentration), ("--path-length", args.path_length)):
        if not np.isfinite(value) or value < 0:
            parser.error(f"{name} must be finite and non-negative")
    if not np.isfinite(args.photons) or args.photons < 0 or args.photons != int(args.photons):
        parser.error("--photons must be a non-negative integer for the Fock comparison")
    args.photons = int(args.photons)
    if args.sweep_max < 0:
        parser.error("--sweep-max must be non-negative")
    if args.trials < 1 or args.seed < 0:
        parser.error("--trials must be positive and --seed must be non-negative")
    if not np.isfinite(args.background_max) or args.background_max < 0:
        parser.error("--background-max must be finite and non-negative")
    if args.experiment in ("unknown", "both"):
        low, high = args.concentration_range
        if not np.all(np.isfinite([low, high])) or not 0 <= low < high or args.concentration_grid < 2:
            parser.error("Use a finite concentration range with 0 <= LOW < HIGH and at least two grid points")
        probes = args.probe_wavenumbers
        if len(probes) < 2 or len(set(probes)) != len(probes) or not np.all(np.isfinite(probes)):
            parser.error("Choose at least two distinct, finite probe wavenumbers")
        if args.true_concentration is not None and not low <= args.true_concentration <= high:
            parser.error("--true-concentration must be inside --concentration-range")
        # Validate probe settings before running either experiment or writing plots.
        for path in (ACETONE_SPECTRUM, ETHANOL_SPECTRUM):
            _, x, a = load_spectrum(path)
            for probe in probes:
                try:
                    transmission_at_wavenumber(x, a, probe, 0, 1)
                except ValueError as error:
                    parser.error(str(error))
    args.output_dir = create_run_directory(args.output_dir)
    brief_fig, brief_summary = None, None
    if args.experiment in ("brief", "both"):
        brief_fig, brief_summary = run_brief_experiment(args)
    if args.experiment == "brief":
        save_run_report(args, {"brief": brief_summary})
        if not args.no_show:
            plt.show()
        plt.close(brief_fig)
        return
    if args.experiment == "unknown":
        unknown_figures, summary = run_unknown_experiment(args)
        save_run_report(args, {"unknown": summary})
        if not args.no_show:
            plt.show()
        for figure in unknown_figures:
            plt.close(figure)
        return

    print(
        f"Transmission model: concentration = {args.concentration:g} micromol/mol; "
        f"path length = {args.path_length:g} m for both molecules."
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    transmission_fig, transmission_ax = plt.subplots(figsize=(10, 5.5))
    names, probe_transmissions = [], []
    for path in (ACETONE_SPECTRUM, ETHANOL_SPECTRUM):
        name, wavenumber, absorption = load_spectrum(path)
        try:
            probe_transmission = transmission_at_wavenumber(
                wavenumber, absorption, args.wavenumber, args.concentration, args.path_length
            )
        except ValueError as error:
            parser.error(str(error))
        names.append(name)
        probe_transmissions.append(probe_transmission)
        ax.plot(wavenumber, absorption, label=name, linewidth=1.2)
        transmission = absorption_to_transmission(
            absorption, args.concentration, args.path_length
        )
        transmission_ax.plot(wavenumber, transmission, label=name, linewidth=1.2)
        print(
            f"Loaded {name}: {len(wavenumber):,} points, "
            f"{wavenumber[0]:.1f} to {wavenumber[-1]:.1f} cm^-1; "
            f"peak absorption coefficient = {absorption.max():.4g}"
        )
        print(
            f"  Modelled transmission: {transmission.min():.2%} to {transmission.max():.2%}; "
            f"{np.count_nonzero(absorption < 0)} negative baseline values treated as zero absorption."
        )

    ax.set_title("Acetone and ethanol: NIST gas-phase infrared spectra")
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Base-10 absorption coefficient\n" + r"[(µmol/mol)$^{-1}$ m$^{-1}$]")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.text(
        0.5, 0.02,
        "Measured reference spectra • Original baseline values retained",
        ha="center", fontsize=9, color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(fig, OUTPUT_PLOT, args.output_dir)
    print(f"Saved comparison plot to: {args.output_dir / OUTPUT_PLOT.name}")

    transmission_ax.set_title(
        "Acetone and ethanol: modelled photon transmission\n"
        f"Concentration = {args.concentration:g} µmol/mol; path length = {args.path_length:g} m"
    )
    transmission_ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    transmission_ax.set_ylabel(r"Photon transmission probability, $\eta$")
    transmission_ax.set_ylim(0, 1.03)
    transmission_ax.grid(True, alpha=0.25)
    transmission_ax.axvline(args.wavenumber, color="gray", linestyle="--",
                            label=f"Probe: {args.wavenumber:g} cm⁻¹")
    transmission_ax.legend()
    transmission_fig.text(
        0.5, 0.02,
        "Ideal absorption model • Negative absorption set to zero for transmission only",
        ha="center", fontsize=9, color="dimgray",
    )
    transmission_fig.tight_layout(rect=(0, 0.05, 1, 1))
    save_figure(transmission_fig, TRANSMISSION_PLOT, args.output_dir)
    print(f"Saved transmission plot to: {args.output_dir / TRANSMISSION_PLOT.name}")

    counted_probabilities = detection_probability(probe_transmissions, args.detector_efficiency)
    expected_counts = args.photons * counted_probabilities
    print(f"\nCoherent-light experiment at {args.wavenumber:g} cm^-1:")
    print(f"Mean incoming photons per measurement: {args.photons:g}")
    print(f"Detector efficiency: {args.detector_efficiency:.2%} (same for both probes)")
    print(f"Mean recorded background: {args.background_counts:g} counts per measurement (same for both probes)")
    for name, eta, probability, mean in zip(names, probe_transmissions, counted_probabilities, expected_counts):
        print(f"  {name}: sample transmission {eta:.4%}; counted probability {probability:.4%}; "
              f"mean detected signal {mean:.2f}; mean recorded total {mean + args.background_counts:.2f}")
    true_index = None if args.true_molecule is None else ("acetone", "ethanol").index(args.true_molecule)
    result = simulate_identification(expected_counts, args.trials, args.seed, true_index,
                                     background_counts=args.background_counts)
    print(f"\nOne hidden-sample example: recorded {result['observed']} counts (signal + background)")
    print(f"  Classifier prediction: {names[result['predicted']]}")
    print(f"  True simulated sample (revealed afterwards): {names[result['true_index']]}")
    print(f"  Decision: {'correct' if result['predicted'] == result['true_index'] else 'incorrect'}")
    print(f"\nRepeated trials: {args.trials:,} measurements per molecule")
    for name, row in zip(names, result["confusion"]):
        print(f"  True {name}: {row[0]:,} predicted {names[0]}, {row[1]:,} predicted {names[1]}")
    print(f"Equal-prior simulated accuracy: {result['accuracy']:.2%}")
    print(f"Equal-prior simulated error: {1 - result['accuracy']:.2%}")
    if expected_counts[0] == expected_counts[1]:
        print("The models are identical: counts contain no identity information (chance accuracy 50%).")
    print("Exact likelihood ties choose Acetone. These results test simulated samples, not real unknowns.")
    identification_fig = plot_identification(
        names, expected_counts, result, args.wavenumber, args.photons,
        detector_efficiency=args.detector_efficiency,
        background_counts=args.background_counts, output_dir=args.output_dir,
    )
    print(f"Saved identification plot to: {args.output_dir / IDENTIFICATION_PLOT.name}")
    # A separate reproducible stream; compare the same hidden identity in the example.
    fock_result = simulate_identification(
        expected_counts, args.trials, np.random.SeedSequence([args.seed, 1]),
        result["true_index"], fock_photons=args.photons,
        background_counts=args.background_counts,
    )
    print(f"\nFock-state experiment: exactly {args.photons} incoming photons per measurement")
    print(f"  Example: {fock_result['observed']} recorded counts; predicted {names[fock_result['predicted']]}; "
          f"true sample {names[fock_result['true_index']]}")
    for name, row in zip(names, fock_result["confusion"]):
        print(f"  True {name}: {row[0]:,} predicted {names[0]}, {row[1]:,} predicted {names[1]}")
    print(f"Fock simulated accuracy: {fock_result['accuracy']:.4%}")
    print(f"Fock simulated error: {1 - fock_result['accuracy']:.4%}")
    coherent_error = theoretical_error(expected_counts, background_counts=args.background_counts)
    fock_error = theoretical_error(expected_counts, args.photons, args.background_counts)
    print(f"Theoretical photon-counting error: coherent {coherent_error:.6%}; Fock {fock_error:.6%}")
    if fock_result["accuracy"] == 1:
        print("No Fock errors observed in these finite trials; this does not imply zero theoretical error.")
    print("Comparison assumes ideal Fock-state preparation, known efficiency and independent Poisson background.")
    fock_fig = plot_identification(
        names, expected_counts, fock_result, args.wavenumber, args.photons, fock_photons=args.photons,
        detector_efficiency=args.detector_efficiency,
        background_counts=args.background_counts, output_dir=args.output_dir,
    )
    comparison_fig = plot_probe_comparison(
        probe_transmissions, args.photons, args.sweep_max, result, fock_result,
        args.wavenumber, args.concentration, args.path_length, args.detector_efficiency,
        args.background_counts, output_dir=args.output_dir,
    )
    detector_fig = plot_detector_comparison(
        probe_transmissions, args.photons, args.detector_efficiency, result, fock_result,
        args.wavenumber, args.concentration, args.path_length, args.background_counts, output_dir=args.output_dir,
    )
    background_fig = plot_background_comparison(
        expected_counts, args.photons, args.background_counts, args.background_max,
        result, fock_result, args.wavenumber, args.detector_efficiency,
        args.concentration, args.path_length, output_dir=args.output_dir,
    )
    print("\nTheoretical error as detector efficiency changes (fixed incoming photon budget):")
    efficiencies = [0.0, 0.5, 0.9, 1.0]
    coherent_errors, fock_errors = detector_error_curve(
        probe_transmissions, args.photons, efficiencies, args.background_counts
    )
    for efficiency, classical, quantum in zip(efficiencies, coherent_errors, fock_errors):
        print(f"  Efficiency {efficiency:.0%}: coherent {classical:.6%}; Fock {quantum:.6%}")
    print("\nTheoretical error as recorded background increases (other settings fixed):")
    levels = np.unique([0, args.background_counts, 10, args.background_max])
    coherent_errors, fock_errors = background_error_curve(expected_counts, args.photons, levels)
    for level, coherent, fock in zip(levels, coherent_errors, fock_errors):
        print(f"  Mean background {level:g}: coherent {coherent:.6%}; Fock {fock:.6%}")
    print(f"Saved Fock results to: {args.output_dir / FOCK_PLOT.name}")
    print(f"Saved error comparison to: {args.output_dir / COMPARISON_PLOT.name}")
    print(f"Saved detector efficiency comparison to: {args.output_dir / DETECTOR_PLOT.name}")
    print(f"Saved background comparison to: {args.output_dir / BACKGROUND_PLOT.name}")
    experiments = {"known": {
        "molecule_order": names, "transmissions": probe_transmissions,
        "coherent": {**result, "theoretical_error": coherent_error},
        "fock": {**fock_result, "theoretical_error": fock_error},
    }}
    unknown_figures = []
    if brief_summary is not None:
        experiments["brief"] = brief_summary
    if args.experiment == "both":
        unknown_figures, experiments["unknown"] = run_unknown_experiment(args)
    save_run_report(args, experiments)
    if not args.no_show:
        print("Displaying the plots. Close all plot windows to finish.")
        plt.show()
    plt.close(fig)
    plt.close(transmission_fig)
    plt.close(identification_fig)
    plt.close(fock_fig)
    plt.close(comparison_fig)
    plt.close(detector_fig)
    plt.close(background_fig)
    if brief_fig is not None:
        plt.close(brief_fig)
    for figure in unknown_figures:
        plt.close(figure)


if __name__ == "__main__":
    main()
