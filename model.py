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

# Step 3 - gradient_histograms
def gradient_histograms(codes, g, h, n_bins):
    """
    Sum gradients and Hessians by feature and bin.

    Parameters
    ----------
    codes : ndarray of shape (n, p)
        Integer bin codes for n examples and p features.

    g : ndarray of shape (n,)
        Per-example gradients.

    h : ndarray of shape (n,)
        Per-example Hessians.

    n_bins : int
        Number of bins.

    Returns
    -------
    G : ndarray of shape (p, n_bins)
        Sum of gradients for each feature and bin.

    H : ndarray of shape (p, n_bins)
        Sum of Hessians for each feature and bin.
    """
    codes = np.asarray(codes)
    g = np.asarray(g, dtype=float)
    h = np.asarray(h, dtype=float)

    n, p = codes.shape

    if g.shape != (n,) or h.shape != (n,):
        raise ValueError("g and h must have length equal to the number of rows in codes.")

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("codes must contain integer bin codes.")

    if codes.size and (np.any(codes < 0) or np.any(codes >= n_bins)):
        raise ValueError("Bin codes must be between 0 and n_bins - 1.")

    # Initialize gradient and Hessian histogram arrays.
    G = np.zeros((p, n_bins), dtype=float)
    H = np.zeros((p, n_bins), dtype=float)

    # Aggregate gradients and Hessians separately for each feature.
    # np.bincount performs the summation without Python loops over rows.
    for j in range(p):
        G[j] = np.bincount(
            codes[:, j],
            weights=g,
            minlength=n_bins
        )

        H[j] = np.bincount(
            codes[:, j],
            weights=h,
            minlength=n_bins
        )

    return G, H

# Step 4 - best_split
def best_split(G, H, lam=0.0, gamma=0.0, min_child_weight=1e-3):
    """
    Find the best split using gradient and Hessian histograms.

    Parameters
    ----------
    G : ndarray of shape (p, n_bins)
        Gradient sums for each feature and bin.

    H : ndarray of shape (p, n_bins)
        Hessian sums for each feature and bin.

    lam : float
        L2 regularization parameter.

    gamma : float
        Minimum gain required to justify a split.

    min_child_weight : float
        Minimum total Hessian required in each child.

    Returns
    -------
    tuple or None
        (feature, t, gain) for the best split, or None if no
        worthwhile split is available.
    """
    G = np.asarray(G, dtype=float)
    H = np.asarray(H, dtype=float)

    if G.ndim != 2 or H.ndim != 2:
        raise ValueError("G and H must be 2D arrays.")

    if G.shape != H.shape:
        raise ValueError("G and H must have the same shape.")

    p, n_bins = G.shape

    # A split is impossible if there is only one bin.
    if n_bins <= 1:
        return None

    best_gain = -np.inf
    best_feature = None
    best_t = None

    # Examine features in ascending order.
    for j in range(p):

        # Compute cumulative gradient and Hessian sums.
        # The cumulative sums allow efficient evaluation of every threshold.
        G_left = np.cumsum(G[j])
        H_left = np.cumsum(H[j])

        # Total gradient and Hessian sums for this feature.
        G_total = G_left[-1]
        H_total = H_left[-1]

        # Examine all valid split positions.
        for t in range(n_bins - 1):

            # Statistics for the left child (bins 0 through t).
            GL = G_left[t]
            HL = H_left[t]

            # Statistics for the right child (bins t+1 onward).
            GR = G_total - GL
            HR = H_total - HL

            # Both children must satisfy the minimum Hessian requirement.
            if HL < min_child_weight or HR < min_child_weight:
                continue

            # Calculate the gain using the previously implemented function.
            gain = split_gain(
                GL, HL,
                GR, HR,
                lam=lam,
                gamma=gamma
            )

            # Use strict comparison so ties retain the first candidate:
            # lowest feature index, then lowest threshold index.
            if gain > best_gain:
                best_gain = float(gain)
                best_feature = j
                best_t = t

    # Reject splits that do not provide a meaningful improvement.
    if best_feature is None or best_gain <= 1e-12:
        return None

    return (int(best_feature), int(best_t), float(best_gain))

