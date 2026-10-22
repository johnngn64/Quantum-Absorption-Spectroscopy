"""Spectral loading, photon-count models, simulation and molecule fitting.

This module performs no plotting or command-line work when imported.
"""

from pathlib import Path
import re

import numpy as np
from scipy.stats import binom, poisson
from scipy.special import logsumexp

DATA_DIR = Path(__file__).resolve().parent / "data"

ACETONE_SPECTRUM = DATA_DIR / "acetone_nist.jdx"

ETHANOL_SPECTRUM = DATA_DIR / "ethanol_nist.jdx"

UNCERTIFIED_REGIONS = ((1325, 1900), (3550, 3950), (2050, 2225), (2295, 2385))

NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[Ee][+-]?\d+)?")

def load_spectrum(path):
    """Read the plain numeric XYDATA format used by our two NIST files.

    This is not a general reader for compressed JCAMP-DX files. Wavenumbers
    are reconstructed from FIRSTX, LASTX and NPOINTS; the rounded row labels
    and DELTAX are not used to construct the grid. YFACTOR converts the stored
    integers to base-10 absorption coefficients. Negative values are retained.
    """
    path = Path(path)
    metadata = {}
    values = []
    reading_data = False

    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.split("$$", 1)[0].strip()
        if not line:
            continue
        if line.startswith("##"):
            key, _, value = line[2:].partition("=")
            key = key.strip().upper()
            metadata[key] = value.strip()
            reading_data = key == "XYDATA"
            if reading_data and value.strip() != "(X++(Y..Y))":
                raise ValueError(f"{path.name}: unsupported XYDATA format")
        elif reading_data:
            if NUMBER.sub("", line).strip():
                raise ValueError(
                    f"{path.name}, line {line_number}: unsupported numeric encoding"
                )
            numbers = NUMBER.findall(line)
            if len(numbers) < 2:
                raise ValueError(f"{path.name}, line {line_number}: missing data")
            # The first number labels the row's x position; the rest are y values.
            values.extend(float(number) for number in numbers[1:])

    required = ("TITLE", "XUNITS", "YUNITS", "FIRSTX", "LASTX", "NPOINTS", "YFACTOR")
    for key in required:
        if key not in metadata:
            raise ValueError(f"{path.name}: missing {key}")
    if metadata["XUNITS"] != "cm-1":
        raise ValueError(f"{path.name}: expected wavenumbers in cm-1")
    if metadata["YUNITS"] != "(micromol/mol)-1m-1 (base 10)":
        raise ValueError(f"{path.name}: unexpected absorption coefficient units")

    point_count = int(metadata["NPOINTS"])
    if point_count < 2 or len(values) != point_count:
        raise ValueError(
            f"{path.name}: expected {point_count} points, found {len(values)}"
        )
    wavenumber = np.linspace(
        float(metadata["FIRSTX"]), float(metadata["LASTX"]), point_count
    )
    absorption = np.asarray(values) * float(metadata["YFACTOR"])
    if not np.all(np.isfinite(wavenumber)) or not np.all(np.isfinite(absorption)):
        raise ValueError(f"{path.name}: spectrum contains non-finite values")
    if not np.all(np.diff(wavenumber) > 0):
        raise ValueError(f"{path.name}: expected increasing wavenumbers")
    return metadata["TITLE"], wavenumber, absorption

def absorption_to_transmission(absorption, concentration, path_length):
    """Return photon survival probability for an ideal absorbing sample.

    Absorption coefficients are base 10, concentration is in micromol/mol,
    and path length is in metres. Negative measured baseline values are
    treated as zero absorption here; the original spectrum is not modified.
    The model assumes linear absorbance and excludes other optical losses.
    """
    for name, value in (("concentration", concentration), ("path length", path_length)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    absorption = np.asarray(absorption, dtype=float)
    if not np.all(np.isfinite(absorption)):
        raise ValueError("Absorption coefficients must be finite")

    physical_absorption = np.maximum(absorption, 0.0)
    return 10.0 ** (-physical_absorption * concentration * path_length)

def transmission_at_wavenumber(wavenumber, absorption, probe, concentration, path_length):
    """Interpolate a reference coefficient for a simplified monochromatic probe.

    This first simulation uses one spectral point. Comparison with real spectra
    should account for the instrument bandwidth and integrated spectral features.
    """
    if not np.isfinite(probe) or not wavenumber[0] <= probe <= wavenumber[-1]:
        raise ValueError("Probe wavenumber must lie within both reference spectra")
    if any(low <= probe <= high for low, high in UNCERTIFIED_REGIONS):
        raise ValueError("Probe wavenumber lies in a NIST uncertified interval")
    coefficient = np.interp(probe, wavenumber, absorption)
    return float(absorption_to_transmission(coefficient, concentration, path_length))

class FockWithBackground:
    """Independent Binomial(N, p) signal plus Poisson(b) recorded background.

    PMFs are computed by direct convolution, not by replacing the random
    background with a constant offset. Cached values include every possible
    contribution to each requested count. Log-space fallback handles tiny tails.
    """

    def __init__(self, photons, probability, background):
        self.photons = photons
        self.probability = probability
        self.background = background
        self._pmf = np.array([])

    def _ensure_pmf(self, maximum):
        if len(self._pmf) <= maximum:
            signal = binom.pmf(np.arange(min(self.photons, maximum) + 1),
                              self.photons, self.probability)
            noise = poisson.pmf(np.arange(maximum + 1), self.background)
            self._pmf = np.convolve(signal, noise)[:maximum + 1]

    def pmf(self, counts):
        counts = np.asarray(counts, dtype=float)
        valid = np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))
        result = np.zeros(counts.shape)
        if np.any(valid):
            indices = counts[valid].astype(int)
            self._ensure_pmf(int(indices.max()))
            result[valid] = self._pmf[indices]
        return result

    def logpmf(self, counts):
        counts = np.asarray(counts)
        probabilities = self.pmf(counts)
        with np.errstate(divide="ignore"):
            result = np.log(probabilities)
        # The fallback also distinguishes impossible counts from underflow.
        for count in np.unique(counts[(probabilities == 0) & (counts >= 0) &
                                     np.isfinite(counts) & (counts == np.floor(counts))]):
            signal_counts = np.arange(min(self.photons, int(count)) + 1)
            value = logsumexp(binom.logpmf(signal_counts, self.photons, self.probability) +
                              poisson.logpmf(count - signal_counts, self.background))
            result = np.where(counts == count, value, result)
        return result

    def ppf(self, quantile):
        if not 0 < quantile < 1:
            raise ValueError("Quantile must be strictly between 0 and 1")
        # Signal cannot exceed N, so this is a conservative initial upper bound.
        maximum = self.photons + int(poisson.ppf(1 - (1 - quantile) / 2, self.background))
        self._ensure_pmf(maximum)
        cumulative = np.cumsum(self._pmf)
        index = int(np.searchsorted(cumulative, quantile))
        if index >= len(cumulative):
            raise ArithmeticError("Could not resolve the requested count quantile")
        return index

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        return (rng.binomial(self.photons, self.probability, size=size) +
                rng.poisson(self.background, size=size))

    def mean(self):
        return self.photons * self.probability + self.background

    def var(self):
        return self.photons * self.probability * (1 - self.probability) + self.background

def count_models(expected_counts, fock_photons=None, background_counts=0.0):
    """Build Poisson models, or Binomial(N, eta) models for an N-photon Fock probe.

    Expected counts here are SIGNAL ONLY. Background is a known mean number of
    already-recorded extra counts per measurement, equal for both hypotheses.
    It is not multiplied by detector efficiency again.
    Here eta is the total probability of a signal photon being counted: sample
    transmission times detector efficiency. Both probes have mean N * eta.
    Their variances differ: coherent: N * eta; Fock: N * eta * (1 - eta).
    """
    means = np.asarray(expected_counts, dtype=float)
    if means.shape != (2,) or not np.all(np.isfinite(means)) or np.any(means < 0):
        raise ValueError("Expected counts must be two finite, non-negative means")
    if not np.isfinite(background_counts) or background_counts < 0:
        raise ValueError("Background counts must be a finite, non-negative mean")
    if fock_photons is None:
        return [poisson(mean + background_counts) for mean in means]
    if not isinstance(fock_photons, (int, np.integer)) or fock_photons < 0:
        raise ValueError("A Fock probe requires a non-negative integer photon number")
    if np.any(means > fock_photons):
        raise ValueError("Mean detected counts cannot exceed the fixed incident photon number")
    probabilities = means / fock_photons if fock_photons else np.zeros(2)
    if background_counts > 0:
        return [FockWithBackground(fock_photons, eta, background_counts) for eta in probabilities]
    return [binom(fock_photons, eta) for eta in probabilities]

def detection_probability(transmissions, efficiency):
    """Probability that an incident photon survives the sample AND is counted.

    Missed detections are independent Bernoulli losses. The same known efficiency
    applies to both samples and probes. Background is added separately.
    """
    if not np.isfinite(efficiency) or not 0 <= efficiency <= 1:
        raise ValueError("Detector efficiency must be finite and between 0 and 1")
    transmissions = np.asarray(transmissions, dtype=float)
    if (not np.all(np.isfinite(transmissions)) or
            np.any((transmissions < 0) | (transmissions > 1))):
        raise ValueError("Sample transmission probabilities must be between 0 and 1")
    return transmissions * efficiency

def classify_counts(observed_counts, expected_counts, fock_photons=None, background_counts=0.0):
    """Choose the more likely count model with equal prior probability.

    Returns candidate indices (0 or 1). The classifier has no true-label input.
    Exact likelihood ties choose candidate 0; ties contain no identity evidence.
    """
    counts = np.asarray(observed_counts)
    models = count_models(expected_counts, fock_photons, background_counts)
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("Observed counts must be finite, non-negative integers")
    log_likelihoods = np.stack([model.logpmf(counts) for model in models], axis=-1)
    if np.any(np.all(np.isneginf(log_likelihoods), axis=-1)):
        raise ValueError("Observed count is impossible under both models")
    return np.argmax(log_likelihoods, axis=-1)

def simulate_identification(expected_counts, trials, seed, true_index=None, fock_photons=None,
                            background_counts=0.0):
    """Run one hidden-sample example and equally many trials for each molecule.

    Each trial is one fresh measurement with mean incident photon number N_in.
    Add independent Poisson background to Poisson or binomial signal counts.
    Both candidates have known, equal concentration and path length; the means
    already include detector efficiency. Simulating from the same references tests this
    model, not identification of independently measured unknown samples.
    """
    if not isinstance(trials, (int, np.integer)) or trials < 1:
        raise ValueError("Trials must be a positive integer")
    if true_index not in (None, 0, 1):
        raise ValueError("True sample index must be 0 or 1")
    models = count_models(expected_counts, fock_photons, background_counts)
    rng = np.random.default_rng(seed)
    if true_index is None:
        true_index = int(rng.integers(2))
    observed = int(models[true_index].rvs(random_state=rng))
    predicted = int(classify_counts(observed, expected_counts, fock_photons, background_counts))

    confusion = np.zeros((2, 2), dtype=int)
    for actual_index, model in enumerate(models):
        counts = model.rvs(size=trials, random_state=rng)
        decisions = classify_counts(counts, expected_counts, fock_photons, background_counts)
        confusion[actual_index] = np.bincount(decisions, minlength=2)
    return {
        "observed": observed, "predicted": predicted, "true_index": true_index,
        "confusion": confusion, "accuracy": float(np.trace(confusion) / confusion.sum()),
    }

def wilson_interval(successes, total, z=1.959964):
    """Wilson score interval for a binomial proportion (default 95%).

    Returns (low, high) as fractions. Unlike the normal approximation, it stays
    inside [0, 1] and remains informative when no successes or failures occur.
    """
    if not isinstance(total, (int, np.integer)) or total < 1:
        raise ValueError("Total must be a positive integer")
    if not 0 <= successes <= total:
        raise ValueError("Successes must lie between zero and the total")
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)

def theoretical_error(expected_counts, fock_photons=None, background_counts=0.0):
    """Equal-prior photon-counting error: half the overlap of the count PMFs.

    Sum the full binomial support when background is zero. Otherwise truncate the
    upper tail with omitted probability at most approximately 1e-12 per model.
    This is optimal for this count measurement, not an optimization over every
    possible quantum measurement of coherent light.
    """
    models = count_models(expected_counts, fock_photons, background_counts)
    if expected_counts[0] == expected_counts[1]:
        return 0.5
    high = (fock_photons if fock_photons is not None and background_counts == 0 else
            max(int(model.ppf(1 - 1e-12)) for model in models))
    # Chunking avoids allocating the whole support for larger budgets.
    overlap = 0.0
    for start in range(0, high + 1, 100000):
        counts = np.arange(start, min(high + 1, start + 100000))
        overlap += np.minimum(models[0].pmf(counts), models[1].pmf(counts)).sum()
    return float(0.5 * overlap)

def detector_error_curve(transmissions, photons, efficiencies, background_counts=0.0):
    """Hold incident photon budget and sample properties fixed as efficiency varies."""
    # Validate the fixed photon budget even at zero transmission or efficiency.
    count_models([0, 0], photons, background_counts)
    coherent_errors, fock_errors = [], []
    for efficiency in efficiencies:
        means = photons * detection_probability(transmissions, efficiency)
        coherent_errors.append(theoretical_error(means, background_counts=background_counts))
        fock_errors.append(theoretical_error(means, photons, background_counts))
    return np.asarray(coherent_errors), np.asarray(fock_errors)

def background_error_curve(expected_counts, photons, background_levels):
    """Compare probes with fixed signal conditions and increasing recorded noise."""
    coherent, fock = [], []
    for background in background_levels:
        coherent.append(theoretical_error(expected_counts, background_counts=background))
        fock.append(theoretical_error(expected_counts, photons, background))
    return np.asarray(coherent), np.asarray(fock)

def allocate_photons(total, frequencies):
    """Split one incident budget equally, assigning remainder photons in order."""
    if not isinstance(total, (int, np.integer)) or total < 0 or frequencies < 2:
        raise ValueError("Use a non-negative integer budget and at least two frequencies")
    budgets = np.full(frequencies, total // frequencies, dtype=int)
    budgets[:total % frequencies] += 1
    return budgets

def fit_unknown_sample(counts, coefficients, concentrations, budgets, path_length,
                       efficiency, background, fock=False, return_likelihoods=False):
    """Joint maximum-likelihood molecule/concentration fit on a fixed grid.

    Independent frequencies contribute summed log likelihoods. The SAME fitted
    concentration applies to all frequencies. No true labels or concentrations
    are accepted here. This is a profile-likelihood decision, not a posterior
    probability or a confidence interval. Equal molecule scores choose index 0.
    """
    counts = np.atleast_2d(np.asarray(counts))
    coefficients = np.asarray(coefficients, dtype=float)
    concentrations = np.asarray(concentrations, dtype=float)
    budgets = np.asarray(budgets)
    if (budgets.ndim != 1 or len(budgets) < 2 or not np.all(np.isfinite(budgets)) or
            np.any(budgets < 0) or np.any(budgets != np.floor(budgets))):
        raise ValueError("Frequency budgets must be non-negative integers for at least two frequencies")
    if (coefficients.shape != (2, len(budgets)) or not np.all(np.isfinite(coefficients)) or
            np.any(coefficients < 0)):
        raise ValueError("Provide two non-negative absorption coefficients per frequency")
    if (concentrations.ndim != 1 or len(concentrations) < 2 or
            not np.all(np.isfinite(concentrations)) or np.any(concentrations < 0) or
            np.any(np.diff(concentrations) <= 0)):
        raise ValueError("Concentration grid must be finite, non-negative and strictly increasing")
    if not np.isfinite(path_length) or path_length < 0:
        raise ValueError("Path length must be finite and non-negative")
    detection_probability([1, 1], efficiency)
    count_models([0, 0], background_counts=background)
    if (counts.ndim != 2 or counts.shape[1] != len(budgets) or len(counts) == 0 or
            not np.all(np.isfinite(counts)) or np.any(counts < 0) or
            np.any(counts != np.floor(counts))):
        raise ValueError("Each sample needs one finite non-negative integer count per frequency")

    # Reuse likelihoods for repeated observed counts to keep large trial batches fast.
    unique_counts = [np.unique(counts[:, f], return_inverse=True) for f in range(len(budgets))]
    scores = np.zeros((len(counts), 2, len(concentrations)))
    for molecule in range(2):
        for c_index, concentration in enumerate(concentrations):
            probabilities = 10.0 ** (-coefficients[molecule] * concentration * path_length) * efficiency
            for frequency, budget in enumerate(budgets):
                mean = budget * probabilities[frequency]
                model = count_models([mean, mean], int(budget) if fock else None, background)[0]
                values, inverse = unique_counts[frequency]
                scores[:, molecule, c_index] += model.logpmf(values)[inverse]

    best_scores = scores.max(axis=2)
    if np.any(np.all(np.isneginf(best_scores), axis=1)):
        raise ValueError("A measurement is impossible under every candidate in the concentration grid")
    predicted = best_scores.argmax(axis=1)
    rows = np.arange(len(counts))
    curves = scores[rows, predicted]
    best_indices = curves.argmax(axis=1)
    estimated = concentrations[best_indices].copy()
    # Flat likelihoods do not estimate concentration; do not report the first grid point.
    uninformative = np.all(curves == curves[:, :1], axis=1)
    estimated[uninformative] = np.nan
    tied = best_scores[:, 0] == best_scores[:, 1]
    result = {
        "predicted": predicted, "concentration": estimated, "tied": tied,
        "boundary": ((best_indices == 0) | (best_indices == len(concentrations) - 1)) & ~uninformative,
    }
    if return_likelihoods:
        result["log_likelihoods"] = scores
    return result

def binomial_relative_entropy(photons, probability0, probability1):
    """D(Binomial(N, p0) || Binomial(N, p1)) in nats, for the same N."""
    from scipy.special import rel_entr
    n, p, q = np.broadcast_arrays(np.asarray(photons, dtype=float),
                                 np.asarray(probability0, dtype=float),
                                 np.asarray(probability1, dtype=float))
    if np.any(~np.isfinite(n)) or np.any(n < 0) or np.any(n != np.floor(n)):
        raise ValueError("Fock photon budgets must be finite non-negative integers")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)) or np.any((p < 0) | (p > 1) | (q < 0) | (q > 1)):
        raise ValueError("Binomial probabilities must be finite and between zero and one")
    per_photon = rel_entr(p, q) + rel_entr(1 - p, 1 - q)
    # With no incident photons both distributions are a point mass at zero,
    # even when the per-photon divergence is infinite.
    return np.multiply(n, per_photon, out=np.zeros_like(per_photon), where=n != 0)


def poisson_relative_entropy(mean0, mean1):
    """D(Poisson(mean0) || Poisson(mean1)) in nats, including zero means."""
    from scipy.special import kl_div
    a, b = np.broadcast_arrays(np.asarray(mean0, dtype=float), np.asarray(mean1, dtype=float))
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(a < 0) or np.any(b < 0):
        raise ValueError("Poisson means must be finite and non-negative")
    return kl_div(a, b)


def draw_unknown_counts(coefficients, labels, concentrations, budgets, path_length,
                        efficiency, background, seed, fock=False):
    """Simulate samples, with concentrations drawn independently of the fit grid."""
    rng = np.random.default_rng(seed)
    probabilities = 10.0 ** (-coefficients[labels] * concentrations[:, None] * path_length) * efficiency
    if fock:
        signal = rng.binomial(budgets, probabilities)
        return signal + rng.poisson(background, size=signal.shape)
    return rng.poisson(budgets * probabilities + background)
