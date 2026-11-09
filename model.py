"""
Gradient Boosted Trees from Scratch: AdaBoost, LogitBoost and XGBoost

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - fit_bins
import numpy as np

def fit_bins(X, max_bins=64):
    """
    Compute sorted candidate split thresholds (cuts) for each feature.

    If a feature has at most max_bins distinct values, use the midpoints
    between consecutive distinct values.

    Otherwise, use distinct quantiles at the specified levels, excluding
    any cuts that are not strictly below the feature's maximum.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if max_bins < 1:
        raise ValueError("max_bins must be at least 1.")

    edges = []

    # Quantile levels specified in the instructions.
    quantile_levels = np.linspace(0, 1, max_bins + 1)[1:-1]

    for j in range(X.shape[1]):
        column = X[:, j]
        distinct_values = np.unique(column)

        if len(distinct_values) <= max_bins:
            # Use midpoints between consecutive distinct values.
            cuts = (
                distinct_values[:-1] / 2.0
                + distinct_values[1:] / 2.0
            )
        else:
            # Use distinct quantile values as candidate thresholds.
            cuts = np.unique(np.quantile(column, quantile_levels))

            # Keep only cuts strictly below the column maximum.
            cuts = cuts[cuts < np.max(column)]

        edges.append(np.asarray(cuts))

    return edges


def apply_bins(X, edges):
    """
    Replace every feature value with its integer bin code.

    The code is the number of cuts strictly below the corresponding value.
    """
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if X.shape[1] != len(edges):
        raise ValueError("The number of edge arrays must match the number of features.")

    binned_X = np.empty(X.shape, dtype=int)

    for j, cuts in enumerate(edges):
        # side='left' counts only cuts strictly smaller than the value.
        binned_X[:, j] = np.searchsorted(cuts, X[:, j], side="left")

    return binned_X


def n_bins_of(edges):
    """
    Return the number of bins required across all features.

    This is one more than the maximum number of cuts in any feature.
    """
    return 1 + max((len(cuts) for cuts in edges), default=0)

# Step 2 - split_gain
def leaf_weight(G, H, lam=0.0):
    """
    Calculate the optimal weight of a leaf.

    Formula:
        w* = -G / (H + lam)

    Supports both scalar values and NumPy arrays.
    """
    G = np.asarray(G)
    H = np.asarray(H)

    return -G / (H + lam)


def structure_score(G, H, lam=0.0, gamma=0.0):
    """
    Calculate the structure score contribution of a single leaf.

    Formula:
        score = -0.5 * G^2 / (H + lam) + gamma

    Supports both scalar values and NumPy arrays.
    """
    G = np.asarray(G)
    H = np.asarray(H)

    return -0.5 * (G ** 2) / (H + lam) + gamma


def split_gain(GL, HL, GR, HR, lam=0.0, gamma=0.0):
    """
    Calculate the gain obtained by splitting a node into left and right children.

    Formula:
        gain = 0.5 * (
            GL^2 / (HL + lam)
            + GR^2 / (HR + lam)
            - (GL + GR)^2 / (HL + HR + lam)
        ) - gamma

    Supports both scalar values and NumPy arrays.
    """
    GL = np.asarray(GL)
    HL = np.asarray(HL)
    GR = np.asarray(GR)
    HR = np.asarray(HR)

    gain = 0.5 * (
        (GL ** 2) / (HL + lam)
        + (GR ** 2) / (HR + lam)
        - ((GL + GR) ** 2) / (HL + HR + lam)
    ) - gamma

    return gain

