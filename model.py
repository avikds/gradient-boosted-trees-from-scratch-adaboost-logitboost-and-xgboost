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

# Step 5 - tree_predict
def tree_apply(tree, codes):
    """
    Send binned examples through a decision tree and return their leaf IDs.

    Parameters
    ----------
    tree : dict
        Dictionary containing the following NumPy arrays:
        feature, bin, left, right, value, gain, cover.

    codes : ndarray of shape (n, p)
        Integer-binned feature values for n examples and p features.

    Returns
    -------
    ndarray of shape (n,)
        Integer IDs of the leaf nodes reached by each example.
    """
    codes = np.asarray(codes)

    if codes.ndim != 2:
        raise ValueError("codes must be a 2D array.")

    feature = np.asarray(tree["feature"])
    bins = np.asarray(tree["bin"])
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])

    # Initially, every example starts at the root node (node 0).
    node_ids = np.zeros(codes.shape[0], dtype=int)

    # Traverse until every example has reached a leaf.
    while True:
        # Identify examples currently positioned at internal nodes.
        current_features = feature[node_ids]
        active_rows = np.flatnonzero(current_features != -1)

        # Stop when all examples have reached leaf nodes.
        if len(active_rows) == 0:
            break

        # Get the current node IDs and their splitting features.
        active_nodes = node_ids[active_rows]
        split_features = feature[active_nodes]

        # Compare each example's bin code with its node's threshold.
        go_left = (
            codes[active_rows, split_features] <= bins[active_nodes]
        )

        # Update the node IDs according to the split decisions.
        node_ids[active_rows] = np.where(
            go_left,
            left[active_nodes],
            right[active_nodes]
        )

    return node_ids


def tree_predict(tree, codes):
    """
    Predict values using a decision tree.

    Parameters
    ----------
    tree : dict
        Dictionary representing the decision tree.

    codes : ndarray of shape (n, p)
        Integer-binned feature values.

    Returns
    -------
    ndarray of shape (n,)
        Predicted values from the reached leaf nodes.
    """
    # Find the leaf reached by each example.
    leaf_ids = tree_apply(tree, codes)

    # Return the prediction stored at each reached leaf.
    return np.asarray(tree["value"])[leaf_ids]

# Step 6 - grow_tree
def grow_tree(codes, g, h, n_bins, max_depth=3, lam=0.0,
              gamma=0.0, min_child_weight=1e-3,
              max_features=None, rng=None):
    """
    Grow a decision tree greedily using per-example gradients and Hessians.

    Parameters
    ----------
    codes : ndarray of shape (n, p)
        Binned feature values.

    g : ndarray of shape (n,)
        Per-example gradients.

    h : ndarray of shape (n,)
        Per-example Hessians.

    n_bins : int
        Number of bins for each feature.

    max_depth : int
        Maximum tree depth. The root has depth zero.

    lam : float
        L2 regularization parameter.

    gamma : float
        Split penalty.

    min_child_weight : float
        Minimum total Hessian required in each child.

    max_features : int or None
        Number of randomly selected features considered at each node.
        If None, all features are considered.

    rng : random generator or None
        Random number generator used for feature subsampling.

    Returns
    -------
    dict
        Tree represented by equal-length NumPy arrays:
        feature, bin, left, right, value, gain, cover.
    """
    codes = np.asarray(codes)
    g = np.asarray(g, dtype=float)
    h = np.asarray(h, dtype=float)

    if codes.ndim != 2:
        raise ValueError("codes must be a 2D array.")

    n, p = codes.shape

    if n == 0:
        raise ValueError("codes must contain at least one example.")

    if g.shape != (n,) or h.shape != (n,):
        raise ValueError("g and h must have length equal to the number of rows in codes.")

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")

    if not isinstance(max_depth, (int, np.integer)) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")

    if max_features is not None:
        if (
            not isinstance(max_features, (int, np.integer))
            or isinstance(max_features, (bool, np.bool_))
        ):
            raise ValueError("max_features must be a positive integer or None.")

        if max_features < 1 or max_features > p:
            raise ValueError("max_features must be between 1 and the number of features.")

    # Initialize the random number generator when feature subsampling is used.
    if max_features is not None and rng is None:
        rng = np.random.default_rng()
    elif isinstance(rng, (int, np.integer)):
        rng = np.random.default_rng(rng)

    # Lists holding the information for every tree node.
    feature_nodes = []
    bin_nodes = []
    left_nodes = []
    right_nodes = []
    value_nodes = []
    gain_nodes = []
    cover_nodes = []

    def build_node(rows, depth):
        """
        Recursively construct a node and its descendants.

        rows contains the indices of examples belonging to this node.
        """

        node_id = len(feature_nodes)

        # Calculate the gradient and Hessian sums for this node.
        G_sum = np.sum(g[rows])
        H_sum = np.sum(h[rows])

        # Every node is initially created as a leaf.
        # A valid split will update its feature and child information.
        feature_nodes.append(-1)
        bin_nodes.append(-1)
        left_nodes.append(-1)
        right_nodes.append(-1)

        # Store the optimal leaf prediction and total Hessian.
        value_nodes.append(float(leaf_weight(G_sum, H_sum, lam)))
        gain_nodes.append(0.0)
        cover_nodes.append(float(H_sum))

        # A node at maximum depth must remain a leaf.
        if depth >= max_depth:
            return node_id

        # Select the features considered by this node.
        if max_features is None:
            feature_indices = np.arange(p)
        else:
            # Draw a fresh subset without replacement at every node.
            # Sorting preserves tie-breaking by original feature index.
            feature_indices = np.sort(
                rng.choice(p, size=max_features, replace=False)
            )

        # Calculate gradient and Hessian histograms for this node.
        G_hist, H_hist = gradient_histograms(
            codes[rows][:, feature_indices],
            g[rows],
            h[rows],
            n_bins
        )

        # Find the best permissible split.
        split = best_split(
            G_hist,
            H_hist,
            lam=lam,
            gamma=gamma,
            min_child_weight=min_child_weight
        )

        # Keep this node as a leaf if no worthwhile split exists.
        if split is None:
            return node_id

        local_feature, t, gain = split

        # Convert the selected feature's local index to its original index.
        real_feature = int(feature_indices[local_feature])

        # Divide the examples according to the selected split.
        go_left = codes[rows, real_feature] <= t

        left_rows = rows[go_left]
        right_rows = rows[~go_left]

        # Store the split information in the current node.
        feature_nodes[node_id] = real_feature
        bin_nodes[node_id] = int(t)
        gain_nodes[node_id] = float(gain)

        # Recursively grow the left and right child nodes.
        left_child = build_node(left_rows, depth + 1)
        right_child = build_node(right_rows, depth + 1)

        # Store the resulting child node IDs.
        left_nodes[node_id] = left_child
        right_nodes[node_id] = right_child

        return node_id

    # Start growing the tree from the root (node 0).
    build_node(np.arange(n), depth=0)

    # Convert the node information into NumPy arrays.
    tree = {
        "feature": np.asarray(feature_nodes, dtype=int),
        "bin": np.asarray(bin_nodes, dtype=int),
        "left": np.asarray(left_nodes, dtype=int),
        "right": np.asarray(right_nodes, dtype=int),
        "value": np.asarray(value_nodes, dtype=float),
        "gain": np.asarray(gain_nodes, dtype=float),
        "cover": np.asarray(cover_nodes, dtype=float)
    }

    return tree

# Step 7 - prune_tree
def prune_tree(tree, gamma):
    """
    Prune a decision tree bottom-up by collapsing unnecessary splits.

    An internal node is collapsed when both of its children are leaves
    after pruning and its stored gain is strictly less than gamma.

    The original tree is never modified. The returned tree contains
    only nodes reachable from the root, with updated child IDs.

    Parameters
    ----------
    tree : dict
        Tree containing feature, bin, left, right, value, gain, cover.

    gamma : float
        Minimum gain required to retain an internal split.

    Returns
    -------
    dict
        A new, compacted tree in the same format.
    """
    feature = np.asarray(tree["feature"])
    bins = np.asarray(tree["bin"])
    left = np.asarray(tree["left"])
    right = np.asarray(tree["right"])
    value = np.asarray(tree["value"])
    gain = np.asarray(tree["gain"])
    cover = np.asarray(tree["cover"])

    n_nodes = len(feature)

    if n_nodes == 0:
        raise ValueError("The tree must contain at least a root node.")

    if not all(
        len(arr) == n_nodes
        for arr in (bins, left, right, value, gain, cover)
    ):
        raise ValueError("All tree arrays must have equal lengths.")

    # ---------------------------------------------------------
    # Pass 1: Determine which internal nodes should be collapsed.
    # This is performed bottom-up, from leaves toward the root.
    # ---------------------------------------------------------

    collapsed = np.zeros(n_nodes, dtype=bool)

    def mark_collapsed(node_id):
        """
        Determine whether a node becomes a leaf after pruning.

        Returns True if the resulting node is a leaf.
        """
        # An existing leaf remains a leaf.
        if feature[node_id] == -1:
            return True

        # Process both children before evaluating the current node.
        left_is_leaf = mark_collapsed(int(left[node_id]))
        right_is_leaf = mark_collapsed(int(right[node_id]))

        # Collapse only when both children are leaves and
        # the current node's gain is strictly below gamma.
        if (
            left_is_leaf
            and right_is_leaf
            and gain[node_id] < gamma
        ):
            collapsed[node_id] = True
            return True

        return False

    # Start bottom-up pruning decisions from the root.
    mark_collapsed(0)

    # ---------------------------------------------------------
    # Pass 2: Rebuild the tree using only reachable nodes.
    # ---------------------------------------------------------

    new_feature = []
    new_bin = []
    new_left = []
    new_right = []
    new_value = []
    new_gain = []
    new_cover = []

    def rebuild(node_id):
        """
        Copy a surviving node and its descendants.

        Returns the node's new ID in the rebuilt tree.
        """
        new_id = len(new_feature)

        # If the node was collapsed, store it as a leaf.
        if collapsed[node_id]:
            new_feature.append(-1)
            new_bin.append(-1)
            new_left.append(-1)
            new_right.append(-1)
            new_value.append(float(value[node_id]))
            new_gain.append(0.0)
            new_cover.append(float(cover[node_id]))

            return new_id

        # Copy the current node's information.
        new_feature.append(int(feature[node_id]))
        new_bin.append(int(bins[node_id]))
        new_left.append(-1)
        new_right.append(-1)
        new_value.append(float(value[node_id]))
        new_gain.append(float(gain[node_id]))
        new_cover.append(float(cover[node_id]))

        # Existing leaves require no further processing.
        if feature[node_id] == -1:
            return new_id

        # Rebuild only the surviving left and right subtrees.
        left_id = rebuild(int(left[node_id]))
        right_id = rebuild(int(right[node_id]))

        # Update child references to their new node IDs.
        new_left[new_id] = left_id
        new_right[new_id] = right_id

        return new_id

    # Rebuild from the root, ensuring that node 0 remains the root.
    rebuild(0)

    # Return the compacted tree as NumPy arrays.
    return {
        "feature": np.asarray(new_feature, dtype=int),
        "bin": np.asarray(new_bin, dtype=int),
        "left": np.asarray(new_left, dtype=int),
        "right": np.asarray(new_right, dtype=int),
        "value": np.asarray(new_value, dtype=float),
        "gain": np.asarray(new_gain, dtype=float),
        "cover": np.asarray(new_cover, dtype=float)
    }

# Step 8 - additive_predict
def additive_predict(model, X, n_trees=None):
    """
    Evaluate an additive tree ensemble.

    Formula:
        f(x) = f0 + sum(scale[m] * tree[m](x))

    Parameters
    ----------
    model : dict
        Ensemble containing:
        edges, f0, trees, scale.

    X : ndarray of shape (n, p)
        Raw input features.

    n_trees : int or None
        Number of initial trees to use.
        If None, all trees are used.

    Returns
    -------
    ndarray of shape (n,)
        Predictions from the additive model.
    """
    X = np.asarray(X)

    # Bin the input features using the model's stored bin thresholds.
    codes = apply_bins(X, model["edges"])

    trees = model["trees"]
    scales = model["scale"]

    # Determine how many trees to use.
    if n_trees is None:
        n_trees = len(trees)

    if n_trees < 0:
        raise ValueError("n_trees must be non-negative.")

    # Start with the model's initial constant prediction.
    predictions = np.full(X.shape[0], model["f0"], dtype=float)

    # Add the contribution of each selected tree.
    for tree, scale in zip(trees[:n_trees], scales[:n_trees]):
        predictions += scale * tree_predict(tree, codes)

    return predictions


def staged_predict(model, X):
    """
    Return predictions after every stage of the additive ensemble.

    Row 0 contains the initial prediction f0.
    Row m contains the prediction after the first m trees.

    Parameters
    ----------
    model : dict
        Ensemble containing:
        edges, f0, trees, scale.

    X : ndarray of shape (n, p)
        Raw input features.

    Returns
    -------
    ndarray of shape (M + 1, n)
        Predictions at every boosting stage.
    """
    X = np.asarray(X)

    # Bin the input features only once.
    codes = apply_bins(X, model["edges"])

    trees = model["trees"]
    scales = model["scale"]

    n = X.shape[0]
    M = len(trees)

    # Allocate space for predictions at all stages.
    predictions = np.empty((M + 1, n), dtype=float)

    # Stage 0: initial constant prediction.
    predictions[0] = model["f0"]

    # Evaluate each tree exactly once.
    for m in range(M):
        tree_prediction = tree_predict(trees[m], codes)

        # Update predictions using the tree's corresponding scale.
        predictions[m + 1] = (
            predictions[m] + scales[m] * tree_prediction
        )

    return predictions

