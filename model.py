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

# Step 9 - l2boost_fit
def l2boost_fit(X, y, M=100, nu=1.0, max_depth=2, max_bins=64):
    """
    Fit a least-squares gradient boosting model.

    Each successive tree is trained on the residuals of the current
    ensemble using squared-error gradient and Hessian statistics.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Training feature matrix.

    y : ndarray of shape (n,)
        Target values.

    M : int
        Number of boosting rounds (trees).

    nu : float
        Learning rate applied to every tree.

    max_depth : int
        Maximum depth of each decision tree.

    max_bins : int
        Maximum number of bins used when discretizing features.

    Returns
    -------
    dict
        Model containing:
        edges       : Feature bin thresholds.
        f0          : Initial prediction (mean of y).
        trees       : List of fitted decision trees.
        scale       : Learning rate for each tree.
        train_mse   : Training MSE after each boosting round.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 1 or len(y) != X.shape[0]:
        raise ValueError("y must be a 1D array with one target per row of X.")

    if len(y) == 0:
        raise ValueError("Training data must contain at least one example.")

    if not isinstance(M, (int, np.integer)) or isinstance(M, (bool, np.bool_)) or M < 0:
        raise ValueError("M must be a non-negative integer.")

    # Fit bin thresholds once and transform the training features.
    edges = fit_bins(X, max_bins=max_bins)
    codes = apply_bins(X, edges)
    nb = n_bins_of(edges)

    # Initial model prediction: mean of the target values.
    f0 = float(np.mean(y))

    # Initialize current predictions with the constant model.
    predictions = np.full(len(y), f0, dtype=float)

    # Store the fitted trees, their learning-rate multipliers,
    # and the training MSE after every boosting round.
    trees = []
    scales = []
    train_mse = []

    # Hessians for squared-error loss are equal to one.
    h = np.ones(len(y), dtype=float)

    # Perform M sequential boosting rounds.
    for m in range(M):

        # Calculate residuals of the current ensemble.
        residuals = y - predictions

        # For squared-error boosting:
        # gradient = -residual and Hessian = 1.
        g = -residuals

        # Fit a tree to the current residuals.
        tree = grow_tree(
            codes,
            g,
            h,
            nb,
            max_depth=max_depth
        )

        # Obtain the predictions of the newly fitted tree.
        tree_predictions = tree_predict(tree, codes)

        # Update the ensemble predictions using the learning rate.
        predictions += nu * tree_predictions

        # Store the fitted tree and its corresponding scale.
        trees.append(tree)
        scales.append(nu)

        # Record training MSE after this boosting round.
        mse = float(np.mean((y - predictions) ** 2))
        train_mse.append(mse)

    # Return the additive model.
    model = {
        "edges": edges,
        "f0": f0,
        "trees": trees,
        "scale": scales,
        "train_mse": train_mse
    }

    return model

# Step 10 - shrinkage_path
def shrinkage_path(X_train, y_train, X_test, y_test, nus, M=200, max_depth=2):
    """
    Measure the effect of different learning rates on test error.

    For each learning rate, fit a least-squares boosting model and
    calculate test MSE at every stage, from 0 to M trees.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Training target values.

    X_test : ndarray
        Test feature matrix.

    y_test : ndarray
        Test target values.

    nus : iterable
        Learning rates to evaluate.

    M : int
        Number of boosting rounds.

    max_depth : int
        Maximum depth of each tree.

    Returns
    -------
    dict
        Maps each learning rate to a dictionary containing:
        best_M    : Number of trees with the lowest test MSE.
        best_mse  : Lowest test MSE.
        final_mse : Test MSE after all M trees.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test, dtype=float)

    results = {}

    # Evaluate each specified learning rate independently.
    for nu in nus:

        # Fit a least-squares boosting model.
        model = l2boost_fit(
            X_train,
            y_train,
            M=M,
            nu=nu,
            max_depth=max_depth
        )

        # Obtain predictions at every boosting stage.
        # Shape: (M + 1, number of test examples).
        predictions = staged_predict(model, X_test)

        # Calculate test MSE for every stage, including stage 0.
        test_mse = np.mean(
            (predictions - y_test[None, :]) ** 2,
            axis=1
        )

        # np.argmin returns the first occurrence when multiple stages
        # have the same minimum test MSE.
        best_M = int(np.argmin(test_mse))

        # Store the best and final test errors.
        results[nu] = {
            "best_M": best_M,
            "best_mse": float(test_mse[best_M]),
            "final_mse": float(test_mse[-1])
        }

    return results

# Step 11 - weighted_stump
def weighted_stump(codes, y, w, n_bins):
    """
    Find the one-split classifier minimizing weighted classification error.

    Parameters
    ----------
    codes : ndarray of shape (n, p)
        Integer-binned feature values.

    y : ndarray of shape (n,)
        Class labels, containing only -1 and +1.

    w : ndarray of shape (n,)
        Non-negative example weights.

    n_bins : int
        Number of bins.

    Returns
    -------
    dict
        Three-node decision tree with keys:
        feature, bin, left, right, value, gain, cover.
    """
    codes = np.asarray(codes)
    y = np.asarray(y)
    w = np.asarray(w, dtype=float)

    if codes.ndim != 2:
        raise ValueError("codes must be a 2D array.")

    n, p = codes.shape

    if y.shape != (n,) or w.shape != (n,):
        raise ValueError("y and w must have length equal to the number of rows in codes.")

    if not isinstance(n_bins, (int, np.integer)) or n_bins < 2:
        raise ValueError("n_bins must be an integer of at least 2.")

    if p == 0:
        raise ValueError("At least one feature is required.")

    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("codes must contain integer bin codes.")

    if codes.size and (np.any(codes < 0) or np.any(codes >= n_bins)):
        raise ValueError("Bin codes must be between 0 and n_bins - 1.")

    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Labels y must contain only -1 and +1.")

    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Weights must be finite and non-negative.")

    if not np.all(np.isfinite(y)):
        raise ValueError("Labels must be finite.")

    # Separate the weights belonging to positive and negative examples.
    positive_weights = w * (y == 1)
    negative_weights = w * (y == -1)

    total_positive = np.sum(positive_weights)
    total_negative = np.sum(negative_weights)

    # There are n_bins - 1 possible split thresholds per feature.
    n_thresholds = n_bins - 1

    # Store errors for both orientations of every candidate split.
    # Orientation 0: -1 on the left, +1 on the right.
    # Orientation 1: +1 on the left, -1 on the right.
    errors = np.empty((p, n_thresholds, 2), dtype=float)

    # Process each feature using weighted histograms.
    for j in range(p):

        # Calculate positive and negative weight histograms.
        pos_hist = np.bincount(
            codes[:, j],
            weights=positive_weights,
            minlength=n_bins
        )

        neg_hist = np.bincount(
            codes[:, j],
            weights=negative_weights,
            minlength=n_bins
        )

        # Cumulative weights for bins 0 through t.
        # Excluding the last bin gives all valid thresholds.
        pos_left = np.cumsum(pos_hist)[:-1]
        neg_left = np.cumsum(neg_hist)[:-1]

        # Orientation 0:
        # Left predicts -1, right predicts +1.
        # Errors = positive weight on left + negative weight on right.
        error_neg_left = (
            pos_left + total_negative - neg_left
        )

        # Orientation 1:
        # Left predicts +1, right predicts -1.
        # Errors = negative weight on left + positive weight on right.
        error_pos_left = (
            neg_left + total_positive - pos_left
        )

        errors[j, :, 0] = error_neg_left
        errors[j, :, 1] = error_pos_left

    # Flattening preserves the required tie-breaking order:
    # lowest feature, then lowest threshold, then -1 on the left.
    best_index = np.argmin(errors)

    feature_idx, threshold_idx, orientation = np.unravel_index(
        best_index, errors.shape
    )

    # Determine the predictions for the two leaves.
    if orientation == 0:
        left_value = -1.0
        right_value = 1.0
    else:
        left_value = 1.0
        right_value = -1.0

    # Construct the required three-node tree.
    stump = {
        "feature": np.array([feature_idx, -1, -1], dtype=int),
        "bin": np.array([threshold_idx, -1, -1], dtype=int),
        "left": np.array([1, -1, -1], dtype=int),
        "right": np.array([2, -1, -1], dtype=int),
        "value": np.array([0.0, left_value, right_value], dtype=float),
        "gain": np.zeros(3, dtype=float),
        "cover": np.zeros(3, dtype=float)
    }

    return stump

# Step 12 - adaboost_fit
def optimal_beta(y, F, w):
    """
    Calculate the weighted classification error and optimal
    AdaBoost step size in half-log-odds units.

    Parameters
    ----------
    y : ndarray
        True labels in {-1, +1}.

    F : ndarray
        Weak learner predictions in {-1, +1}.

    w : ndarray
        Non-negative example weights.

    Returns
    -------
    tuple
        (err, beta), where err is the weighted error and beta
        is the optimal step size.
    """
    y = np.asarray(y)
    F = np.asarray(F)
    w = np.asarray(w, dtype=float)

    if y.ndim != 1 or F.shape != y.shape or w.shape != y.shape:
        raise ValueError("y, F, and w must be 1D arrays of equal length.")

    if len(y) == 0:
        raise ValueError("At least one example is required.")

    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("y must contain only -1 and +1.")

    if not np.all(np.isin(F, [-1, 1])):
        raise ValueError("F must contain only -1 and +1.")

    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("Weights must be finite and non-negative.")

    total_weight = np.sum(w)

    if total_weight <= 0:
        raise ValueError("The sum of weights must be positive.")

    # Calculate the weighted classification error.
    err = float(np.sum(w * (y != F)) / total_weight)

    # Clip only for numerical stability in the logarithm.
    # Return the original, unclipped error.
    clipped_err = np.clip(err, 1e-12, 1.0 - 1e-12)

    # Optimal step size in half-log-odds units.
    beta = 0.5 * np.log(
        (1.0 - clipped_err) / clipped_err
    )

    return err, float(beta)


def adaboost_fit(X, y01, M=100, max_bins=64):
    """
    Fit an AdaBoost.M1 classifier using weighted decision stumps.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Training feature matrix.

    y01 : ndarray of shape (n,)
        Binary labels in {0, 1}.

    M : int
        Maximum number of boosting rounds.

    max_bins : int
        Maximum number of bins for feature discretization.

    Returns
    -------
    dict
        Model containing:
        edges   : Feature bin thresholds.
        f0      : Initial score (0.0).
        trees   : List of fitted weighted stumps.
        scale   : List of optimal beta coefficients.
        errs    : List of weighted classification errors.
        weights : Final normalized example weights.
    """
    X = np.asarray(X)
    y01 = np.asarray(y01)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n = X.shape[0]

    if n == 0 or y01.shape != (n,):
        raise ValueError("y01 must contain one label per training example.")

    if not np.all(np.isin(y01, [0, 1])):
        raise ValueError("y01 must contain only 0 and 1.")

    if not isinstance(M, (int, np.integer)) or isinstance(M, (bool, np.bool_)) or M < 0:
        raise ValueError("M must be a non-negative integer.")

    # Convert labels from {0, 1} to {-1, +1}.
    y = 2.0 * y01.astype(float) - 1.0

    # Fit bin thresholds once for the entire training dataset.
    edges = fit_bins(X, max_bins=max_bins)
    codes = apply_bins(X, edges)
    nb = n_bins_of(edges)

    # Initialize all example weights uniformly.
    w = np.full(n, 1.0 / n, dtype=float)

    # Store fitted weak learners and their coefficients.
    trees = []
    scales = []
    errs = []

    # Perform sequential boosting rounds.
    for m in range(M):

        # 1. Fit the best weighted decision stump.
        stump = weighted_stump(codes, y, w, nb)

        # Obtain predictions from the weak learner.
        F = tree_predict(stump, codes)

        # 2. Calculate weighted error and optimal step size.
        err, beta = optimal_beta(y, F, w)

        # Discard a weak learner that is no better than random guessing.
        if err >= 0.5:
            break

        # Retain the accepted stump and its coefficient.
        trees.append(stump)
        scales.append(beta)
        errs.append(err)

        # Stop immediately if the stump classifies every example correctly.
        if err == 0.0:
            break

        # 3. Update example weights.
        #
        # The model score is f = sum(beta * F).
        # Therefore, Equation 18.33 gives:
        #
        # w_i proportional to exp(-y_i * f_i)
        #
        # The update must use beta, NOT 2 * beta.

        log_w = np.full(n, -np.inf, dtype=float)

        positive_weights = w > 0

        log_w[positive_weights] = (
            np.log(w[positive_weights])
            - beta * y[positive_weights] * F[positive_weights]
        )

        # Numerically stable normalization.
        log_w -= np.max(log_w)

        w = np.exp(log_w)
        w /= np.sum(w)

    # Return the fitted AdaBoost model.
    return {
        "edges": edges,
        "f0": 0.0,
        "trees": trees,
        "scale": scales,
        "errs": errs,
        "weights": w
    }

# Step 13 - probability_report
def half_logit_proba(f):
    """
    Convert half-log-odds scores into probabilities.

    Formula:
        p = 1 / (1 + exp(-2*f))

    Uses a numerically stable implementation to avoid
    overflow for large positive or negative scores.

    Parameters
    ----------
    f : array-like
        Half-log-odds scores.

    Returns
    -------
    ndarray or numpy scalar
        Predicted probabilities.
    """
    f = np.asarray(f, dtype=float)

    # Numerically stable sigmoid transformation.
    # logaddexp avoids overflow for large score magnitudes.
    return np.exp(-np.logaddexp(0.0, -2.0 * f))


def probability_report(y01, p, n_bins=10):
    """
    Evaluate predicted probabilities using log-loss, ECE and
    classification error.

    Parameters
    ----------
    y01 : ndarray of shape (n,)
        True binary labels in {0, 1}.

    p : ndarray of shape (n,)
        Predicted probabilities in [0, 1].

    n_bins : int
        Number of equal-width bins used for ECE.

    Returns
    -------
    dict
        Dictionary containing:
        log_loss : Mean binary cross-entropy in nats.
        ece      : Expected calibration error.
        error    : Classification error rate.
    """
    y01 = np.asarray(y01, dtype=float)
    p = np.asarray(p, dtype=float)

    if y01.ndim != 1 or p.ndim != 1 or y01.shape != p.shape:
        raise ValueError("y01 and p must be 1D arrays of equal length.")

    if len(y01) == 0:
        raise ValueError("At least one example is required.")

    if not np.all(np.isfinite(y01)) or not np.all(np.isin(y01, [0.0, 1.0])):
        raise ValueError("y01 must contain only finite binary labels (0 or 1).")

    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must be finite and lie in [0, 1].")

    if (
        not isinstance(n_bins, (int, np.integer))
        or isinstance(n_bins, (bool, np.bool_))
        or n_bins < 1
    ):
        raise ValueError("n_bins must be a positive integer.")

    n = len(y01)

    # ---------------------------------------------------------
    # 1. Binary cross-entropy (log-loss).
    # ---------------------------------------------------------

    # Clip probabilities to avoid log(0).
    p_clipped = np.clip(p, 1e-12, 1.0 - 1e-12)

    log_loss = -np.mean(
        y01 * np.log(p_clipped)
        + (1.0 - y01) * np.log1p(-p_clipped)
    )

    # ---------------------------------------------------------
    # 2. Expected Calibration Error (ECE).
    # ---------------------------------------------------------

    # Assign predictions to equal-width bins in [0, 1].
    # The minimum operation ensures that p = 1 belongs to
    # the final bin rather than an invalid bin n_bins.
    bin_ids = np.minimum(
        (p * n_bins).astype(int),
        n_bins - 1
    )

    ece = 0.0

    for b in range(n_bins):

        # Identify examples belonging to the current bin.
        mask = bin_ids == b

        # Ignore empty bins.
        if not np.any(mask):
            continue

        # Mean observed accuracy in this bin.
        bin_accuracy = np.mean(y01[mask])

        # Mean predicted confidence in this bin.
        bin_confidence = np.mean(p[mask])

        # Weight calibration error by the fraction of examples.
        bin_fraction = np.sum(mask) / n

        ece += bin_fraction * abs(
            bin_accuracy - bin_confidence
        )

    # ---------------------------------------------------------
    # 3. Classification error.
    # ---------------------------------------------------------

    predicted_labels = (p > 0.5).astype(float)

    error = np.mean(predicted_labels != y01)

    # Return ordinary Python floats.
    return {
        "log_loss": float(log_loss),
        "ece": float(ece),
        "error": float(error)
    }

# Step 14 - working_response
def working_response(y01, f, z_max=4.0):
    """
    Calculate the LogitBoost working response and Hessian weights.

    Parameters
    ----------
    y01 : ndarray
        Binary target labels in {0, 1}.

    f : ndarray
        Current model scores in half-log-odds units.

    z_max : float
        Maximum absolute value allowed for the working response.

    Returns
    -------
    tuple
        (z, w), where:
        z : Clipped working responses.
        w : Hessian weights pi * (1 - pi).

    The implementation avoids dividing by w, ensuring finite
    working responses even for highly confident predictions.
    """
    y01 = np.asarray(y01, dtype=float)
    f = np.asarray(f, dtype=float)

    # Support scalar inputs and broadcast-compatible arrays.
    y01, f = np.broadcast_arrays(y01, f)

    if not np.all(np.isfinite(y01)) or not np.all(np.isin(y01, [0.0, 1.0])):
        raise ValueError("y01 must contain only finite binary labels (0 or 1).")

    if not np.all(np.isfinite(f)):
        raise ValueError("f must contain finite values.")

    if not np.isfinite(z_max) or z_max <= 0:
        raise ValueError("z_max must be a positive finite number.")

    # Limit extreme scores before multiplying by 2, avoiding
    # floating-point overflow for very large finite inputs.
    max_float = np.finfo(float).max
    f_safe = np.clip(f, -max_float / 2.0, max_float / 2.0)

    # Calculate probabilities using the previously implemented function.
    pi = half_logit_proba(f_safe)

    # Calculate Hessian weights.
    w = pi * (1.0 - pi)

    # Calculate the logarithm of the absolute working response.
    # This avoids directly calculating potentially enormous exponentials.
    log_abs_z = np.where(
        y01 == 1.0,
        np.logaddexp(0.0, -2.0 * f_safe),
        np.logaddexp(0.0, 2.0 * f_safe)
    )

    # Apply the maximum allowed absolute response in log space.
    log_abs_z = np.minimum(log_abs_z, np.log(z_max))

    # Recover the bounded absolute working response.
    abs_z = np.exp(log_abs_z)

    # Apply the correct sign according to the binary target.
    z = np.where(y01 == 1.0, abs_z, -abs_z)

    return z, w

# Step 15 - logitboost_fit
def half_logit_loss(y01, f):
    """
    Calculate the mean binary logistic loss in half-log-odds units.

    Formula:
        L = mean(log(1 + exp(-2 * (2*y01 - 1) * f)))

    Uses np.logaddexp for numerical stability.

    Parameters
    ----------
    y01 : ndarray of shape (n,)
        Binary target labels in {0, 1}.

    f : ndarray of shape (n,)
        Model scores in half-log-odds units.

    Returns
    -------
    float
        Mean logistic loss.
    """
    y01 = np.asarray(y01, dtype=float)
    f = np.asarray(f, dtype=float)

    if y01.ndim != 1 or f.shape != y01.shape:
        raise ValueError("y01 and f must be 1D arrays of equal length.")

    if len(y01) == 0:
        raise ValueError("At least one example is required.")

    if not np.all(np.isfinite(y01)) or not np.all(np.isin(y01, [0.0, 1.0])):
        raise ValueError("y01 must contain only finite binary labels (0 or 1).")

    if not np.all(np.isfinite(f)):
        raise ValueError("f must contain finite values.")

    # Convert binary labels to signed labels {-1, +1}.
    y_signed = 2.0 * y01 - 1.0

    # Stable computation of log(1 + exp(x)).
    losses = np.logaddexp(0.0, -2.0 * y_signed * f)

    return float(np.mean(losses))


def logitboost_fit(X, y01, M=100, max_depth=1, nu=1.0,
                   max_bins=64, z_max=4.0):
    """
    Fit a LogitBoost model using Newton-based gradient boosting.

    Each boosting round:
        1. Computes working responses and Hessian weights.
        2. Fits a regression tree using weighted least squares.
        3. Updates the model using nu/2 times the tree predictions.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Training feature matrix.

    y01 : ndarray of shape (n,)
        Binary target labels in {0, 1}.

    M : int
        Number of boosting rounds.

    max_depth : int
        Maximum depth of each regression tree.

    nu : float
        Learning rate.

    max_bins : int
        Maximum number of bins for feature discretization.

    z_max : float
        Maximum absolute value of the working response.

    Returns
    -------
    dict
        Model containing:
        edges      : Feature bin thresholds.
        f0         : Initial score (0.0).
        trees      : List of fitted regression trees.
        scale      : Learning-rate multipliers (nu/2).
        train_loss : Logistic loss after every boosting round.
        weights    : Final normalized Hessian weights.
    """
    X = np.asarray(X)
    y01 = np.asarray(y01, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    n = X.shape[0]

    if n == 0 or y01.shape != (n,):
        raise ValueError("y01 must contain one label per row of X.")

    if not np.all(np.isfinite(y01)) or not np.all(np.isin(y01, [0.0, 1.0])):
        raise ValueError("y01 must contain only finite binary labels (0 or 1).")

    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain finite values.")

    if (
        not isinstance(M, (int, np.integer))
        or isinstance(M, (bool, np.bool_))
        or M < 0
    ):
        raise ValueError("M must be a non-negative integer.")

    if not np.isfinite(nu):
        raise ValueError("nu must be finite.")

    if not np.isfinite(z_max) or z_max <= 0:
        raise ValueError("z_max must be positive and finite.")

    # Fit feature bin thresholds once.
    edges = fit_bins(X, max_bins=max_bins)

    # Convert training features to integer bin codes.
    codes = apply_bins(X, edges)

    # Determine the number of bins.
    nb = n_bins_of(edges)

    # Initialize the model score to zero.
    f = np.zeros(n, dtype=float)

    # Store fitted trees, their scales and training losses.
    trees = []
    scales = []
    train_loss = []

    # Each boosting round performs one Newton-based update.
    for m in range(M):

        # Step 1: Calculate working responses and Hessian weights.
        z, w = working_response(
            y01,
            f,
            z_max=z_max
        )

        # Step 2: Construct gradients and Hessians so that
        # the tree minimizes sum(w * (z - F(x))**2).
        #
        # The leaf weight formula is -G/H.
        g = -w * z
        h = w

        # Fit a real-valued regression tree.
        tree = grow_tree(
            codes,
            g,
            h,
            nb,
            max_depth=max_depth,
            min_child_weight=1e-6
        )

        # Obtain the predictions of the newly fitted tree.
        tree_prediction = tree_predict(tree, codes)

        # Step 3: Apply the Newton boosting update.
        # The factor of 1/2 is required for half-log-odds scores.
        scale = nu / 2.0

        f = f + scale * tree_prediction

        # Store the fitted tree and its corresponding scale.
        trees.append(tree)
        scales.append(float(scale))

        # Record logistic loss after this boosting round.
        train_loss.append(
            half_logit_loss(y01, f)
        )

    # Calculate the Hessian weights at the final model scores.
    _, final_weights = working_response(
        y01,
        f,
        z_max=z_max
    )

    # Normalize final weights to sum to one.
    weight_sum = np.sum(final_weights)

    if weight_sum > 0:
        final_weights = final_weights / weight_sum
    else:
        # Stable fallback if all Hessian weights underflow to zero.
        log_weights = (
            -np.logaddexp(0.0, -2.0 * f)
            -np.logaddexp(0.0, 2.0 * f)
        )

        log_weights -= np.max(log_weights)
        final_weights = np.exp(log_weights)
        final_weights /= np.sum(final_weights)

    # Return the fitted LogitBoost model.
    model = {
        "edges": edges,
        "f0": 0.0,
        "trees": trees,
        "scale": scales,
        "train_loss": train_loss,
        "weights": final_weights
    }

    return model

# Step 16 - label_noise_study
def flip_labels(y01, rate, seed=0):
    """
    Randomly flip a specified fraction of binary labels.

    Parameters
    ----------
    y01 : ndarray of shape (n,)
        Original binary labels in {0, 1}.

    rate : float
        Probability of flipping each label.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    tuple
        noisy : ndarray
            Labels after randomly flipping selected examples.

        flip : ndarray of bool
            Boolean mask indicating which labels were flipped.
    """
    y01 = np.asarray(y01, dtype=float)

    if y01.ndim != 1:
        raise ValueError("y01 must be a 1D array.")

    if not np.all(np.isfinite(y01)) or not np.all(np.isin(y01, [0.0, 1.0])):
        raise ValueError("y01 must contain only binary labels (0 or 1).")

    if not np.isscalar(rate) or not np.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be a number between 0 and 1.")

    # Generate a reproducible random flip mask.
    rng = np.random.default_rng(seed)
    flip = rng.random(len(y01)) < rate

    # Create a copy and invert the selected labels.
    noisy = y01.copy()
    noisy[flip] = 1.0 - noisy[flip]

    return noisy, flip


def label_noise_study(X_train, y_train, X_test, y_test,
                      rates, M=200, seed=0):
    """
    Compare AdaBoost and LogitBoost under different levels of
    training-label noise.

    Both models are trained using the noisy training labels.
    Their performance is evaluated against the clean test labels.

    Parameters
    ----------
    X_train : ndarray
        Training feature matrix.

    y_train : ndarray
        Original clean training labels in {0, 1}.

    X_test : ndarray
        Test feature matrix.

    y_test : ndarray
        Clean test labels in {0, 1}.

    rates : iterable
        Label-flipping probabilities to evaluate.

    M : int
        Number of boosting rounds.

    seed : int
        Random seed used for label flipping.

    Returns
    -------
    dict
        Maps each noise rate to:
        ada_error             : AdaBoost test error.
        logit_error           : LogitBoost test error.
        ada_weight_on_flipped : AdaBoost weight on flipped examples.
        logit_weight_on_flipped : LogitBoost weight on flipped examples.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train, dtype=float)

    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test, dtype=float)

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train and X_test must be 2D arrays.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Training and test data must have the same number of features.")

    if y_train.shape != (X_train.shape[0],):
        raise ValueError("y_train must contain one label per training example.")

    if y_test.shape != (X_test.shape[0],) or len(y_test) == 0:
        raise ValueError("y_test must contain one label per test example.")

    if not np.all(np.isfinite(y_train)) or not np.all(np.isin(y_train, [0.0, 1.0])):
        raise ValueError("y_train must contain only binary labels (0 or 1).")

    if not np.all(np.isfinite(y_test)) or not np.all(np.isin(y_test, [0.0, 1.0])):
        raise ValueError("y_test must contain only binary labels (0 or 1).")

    # Store results for every noise rate.
    results = {}

    # Convert clean test labels into binary classification targets.
    true_labels = y_test > 0.5

    # Evaluate each specified label-noise rate.
    for rate in rates:

        # Generate noisy training labels.
        # The same seed makes the flip masks nested across rates.
        noisy, flip = flip_labels(
            y_train,
            rate,
            seed=seed
        )

        # -----------------------------------------------------
        # 1. Train AdaBoost on noisy training labels.
        # -----------------------------------------------------
        ada_model = adaboost_fit(
            X_train,
            noisy,
            M=M
        )

        # Predict on the clean test data.
        ada_scores = additive_predict(
            ada_model,
            X_test
        )

        ada_predictions = ada_scores > 0.0

        # Calculate AdaBoost test classification error.
        ada_error = float(
            np.mean(ada_predictions != true_labels)
        )

        # Calculate the proportion of final AdaBoost weight
        # assigned to examples whose training labels were flipped.
        ada_weight_on_flipped = float(
            np.sum(ada_model["weights"][flip])
        )

        # -----------------------------------------------------
        # 2. Train LogitBoost on noisy training labels.
        # -----------------------------------------------------
        logit_model = logitboost_fit(
            X_train,
            noisy,
            M=M
        )

        # Predict on the clean test data.
        logit_scores = additive_predict(
            logit_model,
            X_test
        )

        logit_predictions = logit_scores > 0.0

        # Calculate LogitBoost test classification error.
        logit_error = float(
            np.mean(logit_predictions != true_labels)
        )

        # Calculate the proportion of final LogitBoost weight
        # assigned to examples whose training labels were flipped.
        logit_weight_on_flipped = float(
            np.sum(logit_model["weights"][flip])
        )

        # Store the results using plain Python floats.
        results[float(rate)] = {
            "ada_error": ada_error,
            "logit_error": logit_error,
            "ada_weight_on_flipped": ada_weight_on_flipped,
            "logit_weight_on_flipped": logit_weight_on_flipped
        }

    return results

# Step 17 - negative_gradient
def loss_value(loss, y, f, delta=1.0):
    """
    Calculate per-example loss values for five loss functions.

    Supported losses:
        squared     : 0.5 * (y - f)^2
        absolute    : abs(y - f)
        huber       : Huber loss with parameter delta
        exponential : exp(-(2*y - 1)*f)
        logloss     : log(1 + exp(-(2*y - 1)*f))

    Parameters
    ----------
    loss : str
        Name of the loss function.

    y : array-like
        Target values. For exponential and logloss, labels
        must belong to {0, 1}.

    f : array-like
        Model predictions.

    delta : float
        Positive Huber loss threshold.

    Returns
    -------
    ndarray
        Per-example loss values.
    """
    if loss not in ("squared", "absolute", "huber", "exponential", "logloss"):
        raise ValueError(f"Unknown loss: {loss}")

    y, f = np.broadcast_arrays(
        np.asarray(y, dtype=float),
        np.asarray(f, dtype=float)
    )

    if loss in ("exponential", "logloss"):
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("Classification labels must be 0 or 1.")

        # Convert binary labels from {0, 1} to {-1, +1}.
        y_signed = 2.0 * y - 1.0

    if loss == "squared":
        # Squared-error loss.
        r = y - f
        return 0.5 * r**2

    elif loss == "absolute":
        # Absolute-error loss.
        return np.abs(y - f)

    elif loss == "huber":
        # Huber loss.
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be positive and finite.")

        r = y - f
        abs_r = np.abs(r)

        return np.where(
            abs_r <= delta,
            0.5 * r**2,
            delta * (abs_r - 0.5 * delta)
        )

    elif loss == "exponential":
        # Exponential loss for binary classification.
        return np.exp(-y_signed * f)

    elif loss == "logloss":
        # Numerically stable logistic loss.
        # log(1 + exp(x)) = logaddexp(0, x).
        return np.logaddexp(0.0, -y_signed * f)


def negative_gradient(loss, y, f, delta=1.0):
    """
    Calculate the negative gradient of the specified loss
    with respect to the model predictions f.

    Parameters
    ----------
    loss : str
        Name of the loss function.

    y : array-like
        Target values. For exponential and logloss, labels
        must belong to {0, 1}.

    f : array-like
        Model predictions.

    delta : float
        Positive Huber loss threshold.

    Returns
    -------
    ndarray
        Per-example negative gradients.
    """
    if loss not in ("squared", "absolute", "huber", "exponential", "logloss"):
        raise ValueError(f"Unknown loss: {loss}")

    y, f = np.broadcast_arrays(
        np.asarray(y, dtype=float),
        np.asarray(f, dtype=float)
    )

    if loss in ("exponential", "logloss"):
        if not np.all(np.isin(y, [0.0, 1.0])):
            raise ValueError("Classification labels must be 0 or 1.")

        # Convert binary labels from {0, 1} to {-1, +1}.
        y_signed = 2.0 * y - 1.0

    if loss == "squared":
        # Negative gradient of 0.5 * (y - f)^2.
        return y - f

    elif loss == "absolute":
        # Negative gradient of absolute-error loss.
        # At r = 0, use the subgradient 0.
        return np.sign(y - f)

    elif loss == "huber":
        # Negative gradient of Huber loss.
        if not np.isfinite(delta) or delta <= 0:
            raise ValueError("delta must be positive and finite.")

        r = y - f

        return np.clip(r, -delta, delta)

    elif loss == "exponential":
        # Negative gradient of exponential loss.
        return y_signed * np.exp(-y_signed * f)

    elif loss == "logloss":
        # Negative gradient of logistic loss.
        #
        # Equivalent to:
        # y_signed / (1 + exp(y_signed * f))
        #
        # logaddexp provides a numerically stable computation.
        return y_signed * np.exp(
            -np.logaddexp(0.0, y_signed * f)
        )

