"""
Gradient Boosted Trees from Scratch: AdaBoost, LogitBoost and XGBoost scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Gradient Boosted Trees from Scratch (Murphy, Probabilistic Machine Learning, chapter 18).

Runs every method on the book's data: the 100 points of Figure 18.6, then the 4,601 emails of the spam data behind
Figures 18.5, 18.9 and 18.10 on the fixed train/test split from Hastie, Tibshirani and Friedman. Each section prints
the numbers behind one claim in the chapter, including the claims that do not survive."""
import os
import tempfile
import time
import urllib.request

import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from model import (fit_bins, apply_bins, n_bins_of, grow_tree, tree_predict, prune_tree, additive_predict, staged_predict,
                      l2boost_fit, shrinkage_path, adaboost_fit, half_logit_proba, probability_report, logitboost_fit,
                      flip_labels, label_noise_study, loss_value, negative_gradient, gradient_boost_fit, outlier_study,
                      softmax_boost_fit, softmax_boost_proba, xgb_fit, order_study, subsample_study, forest_fit, staged_error,
                      ensemble_error_curves, bias_variance_study, feature_importance, partial_dependence, partial_dependence_2d)

SPAM_NAMES = ("make address all 3d our over remove internet order mail receive will people report addresses free business "
              "email you credit your font 000 money hp hpl george 650 lab labs telnet 857 data 415 85 technology 1999 parts "
              "pm direct cs meeting original project re edu table conference ch; ch( ch[ ch! ch$ ch# CAPAVE CAPMAX CAPTOT").split()


def fetch(url, name):
    path = os.path.join(tempfile.gettempdir(), name)
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=15) as response:
            body = response.read()
        with open(path, "wb") as out:
            out.write(body)
    return path


def load_spam():
    """The 4,601 emails. Rows are the same at UCI and in the ESL files; only ESL publishes the fixed split."""
    for base in ("https://hastie.su.domains/ElemStatLearn/datasets/", "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"):
        try:
            data = np.loadtxt(fetch(base + "spam.data", "spam.data"))
            mask = np.loadtxt(fetch(base + "spam.traintest", "spam.traintest")).astype(int)
            print("spam data: fixed ESL split")
            return data[:, :57], data[:, 57], mask
        except Exception as error:
            print("mirror failed:", type(error).__name__)
    rng = np.random.default_rng(0)
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        data = np.loadtxt(fetch(url, "spambase.data"), delimiter=",")
        print("spam data: UCI copy with a random split of the same size (ESL mirrors unavailable)")
    except Exception as error:
        print("no spam data could be downloaded", type(error).__name__, "- using a seeded stand-in, so ignore the named features below")
        A = rng.exponential(1.0, (4601, 57)) * (rng.random((4601, 57)) < 0.25)
        score = A[:, :6].sum(axis=1) - A[:, 6:12].sum(axis=1) - 0.5
        data = np.c_[A, rng.random(4601) < 1 / (1 + np.exp(-2 * score))].astype(float)
    mask = np.zeros(len(data), dtype=int)
    mask[rng.permutation(len(data))[:1536]] = 1
    return data[:, :57], data[:, 57], mask


def hastie(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    return X, (np.sum(X ** 2, axis=1) > 9.34).astype(float)


def friedman(n, seed, noise=1.0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 10))
    f = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    return X, f + noise * rng.standard_normal(n), f


def error_of(f, y01):
    return float(np.mean((f > 0) != (y01 > 0.5)))


def log_loss_of(f, y01):
    return float(np.mean(np.logaddexp(0.0, -(2 * y01 - 1) * f)))


def section(title):
    print(f"\n=== {title} [{time.time() - START:.0f}s]")


START = time.time()
X, y, mask = load_spam()
X_train, y_train, X_test, y_test = X[mask == 0], y[mask == 0], X[mask == 1], y[mask == 1]
print(f"{len(y_train)} training and {len(y_test)} test emails, {y_train.mean():.1%} spam in training")

# ------------------------------------------------------------------ part 1
section("1. The tree: CART when g = -y and h = 1, and growing before pruning (18.1, 18.5.5.2)")
Xr, yr, _ = friedman(300, 0)
edges = fit_bins(Xr, 100000)
codes, n_bins = apply_bins(Xr, edges), n_bins_of(edges)
ours = tree_predict(grow_tree(codes, -yr, np.ones(300), n_bins, 5), codes)
theirs = DecisionTreeRegressor(max_depth=5).fit(Xr, yr).predict(Xr)
print(f"depth-5 tree vs scikit-learn: largest prediction difference {np.abs(ours - theirs).max():.1e}")
rng = np.random.default_rng(0)
Xx = rng.random((400, 2))
yx = ((Xx[:, 0] > 0.5) ^ (Xx[:, 1] > 0.5)).astype(float)
edges = fit_bins(Xx, 16)
codes, n_bins = apply_bins(Xx, edges), n_bins_of(edges)
g, h = -(yx - yx.mean()), np.ones(400)
print(f"XOR, gamma = 2: stop early {int((grow_tree(codes, g, h, n_bins, 3, gamma=2.0)['feature'] < 0).sum())} leaf, "
      f"grow then prune {int((prune_tree(grow_tree(codes, g, h, n_bins, 3), 2.0)['feature'] < 0).sum())} leaves")

# ------------------------------------------------------------------ part 2
section("2. Least-squares boosting: Figure 18.6 and what shrinkage buys (18.5.2)")
state = np.random.RandomState(42)
xq = state.rand(100, 1) - 0.5
yq = 3 * xq[:, 0] ** 2 + 0.05 * state.randn(100)
grid = np.linspace(-0.5, 0.5, 2001)[:, None]
truth = 3 * grid[:, 0] ** 2
model = l2boost_fit(xq, yq, M=3, nu=1.0, max_depth=2)
stages = np.mean((staged_predict(model, grid) - truth) ** 2, axis=1)
print("Fig 18.6 data, depth-2 trees, MSE against the noiseless curve:",
      ", ".join(f"f{m} {stages[m]:.5f}" for m in range(4)))
Xf, yf, _ = friedman(400, 1)
Xg, _, fg = friedman(4000, 2, noise=0.0)
paths = shrinkage_path(Xf, yf, Xg, fg, [1.0, 0.3, 0.1, 0.03], M=600, max_depth=3)
for nu, row in paths.items():
    print(f"  nu = {nu:<4}  best at {row['best_M']:>3} trees: test MSE {row['best_mse']:.3f}, after 600: {row['final_mse']:.3f}")

# ------------------------------------------------------------------ part 3
section("3. AdaBoost: a stump near chance becomes a strong classifier (18.5, 18.5.3)")
Xh, yh = hastie(2000, 0)
Xv, yv = hastie(10000, 1)
ada_h = adaboost_fit(Xh, yh, M=400)
errors = staged_error(ada_h, Xv, yv)
prod = float(np.prod(2 * np.sqrt(np.array(ada_h["errs"]) * (1 - np.array(ada_h["errs"])))))
print("nested spheres, eq 18.29 stumps, test error:", ", ".join(f"{m} rounds {errors[m - 1]:.4f}" for m in (1, 10, 100, 400)))
print(f"training bound: prod 2 sqrt(err (1-err)) = {prod:.6f}, mean exp loss = {np.mean(np.exp(-(2 * yh - 1) * additive_predict(ada_h, Xh))):.6f}")

section("4. The stumps are the problem, not the loss (18.5.3, 18.5.4)")
real_h = gradient_boost_fit(Xh, yh, loss="exponential", M=400, nu=1.0, max_depth=1)
logit_h = logitboost_fit(Xh, yh, M=400)
library = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=400, random_state=0).fit(Xh, yh)
print("test error after 400 stumps on the same data:")
print(f"  +1/-1 leaves, stump minimizes weighted error (eq 18.29)   {errors[-1]:.4f}")
print(f"  +1/-1 leaves, stump minimizes weighted Gini (scikit-learn) {1 - library.score(Xv, yv):.4f}")
print(f"  real leaves, exponential loss                             {error_of(additive_predict(real_h, Xv), yv):.4f}")
print(f"  real leaves, log loss (LogitBoost)                        {error_of(additive_predict(logit_h, Xv), yv):.4f}")

section("5. Can exponential loss give probabilities? The book says no; eq 18.25 says yes (18.5.3, 18.5.4)")
ada = adaboost_fit(X_train, y_train, M=400)
logit = logitboost_fit(X_train, y_train, M=400)
logit_slow = logitboost_fit(X_train, y_train, M=400, nu=0.1)
print("spam, 400 stumps, probabilities from eq 18.36:")
for name, m in (("AdaBoost", ada), ("LogitBoost", logit), ("LogitBoost nu=0.1", logit_slow)):
    r = probability_report(y_test, half_logit_proba(additive_predict(m, X_test)))
    print(f"  {name:<18} log loss {r['log_loss']:.4f}  ECE {r['ece']:.4f}  error {r['error']:.4f}")

section("6. Mislabeled training data: the mechanism and the consequence (18.5.4)")
study = label_noise_study(X_train, y_train, X_test, y_test, [0.0, 0.05, 0.2], M=200)
for rate, r in study.items():
    print(f"  flip {rate:<4}  error AdaBoost {r['ada_error']:.4f}  LogitBoost {r['logit_error']:.4f}   "
          f"weight on flipped: AdaBoost {r['ada_weight_on_flipped']:.3f}  LogitBoost {r['logit_weight_on_flipped']:.3f}")
print("same trees (depth 3, nu = 0.1, 150 rounds), only the loss differs, clean test error:")
for rate in (0.0, 0.3):
    noisy, _ = flip_labels(y_train, rate, 0)
    row = [error_of(additive_predict(gradient_boost_fit(X_train, noisy, loss=name, M=150, nu=0.1, max_depth=3), X_test), y_test)
           for name in ("exponential", "logloss")]
    print(f"  flip {rate:<4}  exponential {row[0]:.4f}  log loss {row[1]:.4f}")

# ------------------------------------------------------------------ part 5
section("7. Table 18.1 against finite differences (18.5.5)")
rng = np.random.default_rng(1)
yc, fc = rng.integers(0, 2, 500).astype(float), rng.normal(size=500)
signed = 2 * yc - 1
eps = 1e-6
fd_exp = -(loss_value("exponential", yc, fc + eps) - loss_value("exponential", yc, fc - eps)) / (2 * eps)
fd_log = -(loss_value("logloss", yc, fc + eps) - loss_value("logloss", yc, fc - eps)) / (2 * eps)
print(f"exponential row as printed, -y exp(-y f): largest error {np.abs(fd_exp - (-signed * np.exp(-signed * fc))).max():.3f}; "
      f"with the sign fixed: {np.abs(fd_exp - negative_gradient('exponential', yc, fc)).max():.1e}")
print(f"log-loss row with pi = sigma(2f): largest error {np.abs(fd_log - (yc - 1 / (1 + np.exp(-2 * fc)))).max():.3f}; "
      f"with pi = sigma(f): {np.abs(fd_log - negative_gradient('logloss', yc, fc)).max():.1e}")

section("8. Leaf re-solve and robust losses (18.5.5, 18.5.5.1)")
for resolve in (True, False):
    row = [float(np.mean(np.abs(additive_predict(gradient_boost_fit(Xf, scale * yf, loss="absolute", M=100, nu=0.1, resolve=resolve), Xg) - scale * fg))) / scale
           for scale in (1.0, 100.0)]
    print(f"  absolute loss, re-solve {str(resolve):<5}: test MAE / scale, y as is {row[0]:.3f}, y * 100 {row[1]:.3f}")
rng = np.random.default_rng(11)
yo = yf + 30.0 * rng.choice([-1, 1], len(yf)) * (rng.random(len(yf)) < 0.05)
print("5% of the training targets off by 30, RMSE on clean targets after 300 rounds (best stage in brackets):")
for name, delta in (("squared", 1.0), ("absolute", 1.0), ("huber", 1.0), ("huber", 3.0)):
    r = outlier_study(Xf, yo, Xg, fg, [name], M=300, delta=delta)[name]
    print(f"  {name + (' delta=' + str(delta) if name == 'huber' else ''):<16} {r['test_rmse']:.3f} ({r['best_rmse']:.3f})")

section("9. Softmax boosting on handwritten digits (eq 18.42-18.43)")
digits = load_digits()
perm = np.random.default_rng(0).permutation(len(digits.target))
Xd, yd = digits.data[perm], digits.target[perm]
softmax_model = softmax_boost_fit(Xd[:1300], yd[:1300], 10, M=100, nu=0.1, max_depth=3)
for rounds in (1, 10, 100):
    P = softmax_boost_proba(softmax_model, Xd[1300:], rounds)
    print(f"  {rounds:>3} rounds: test accuracy {np.mean(P.argmax(axis=1) == yd[1300:]):.4f}, log loss {-np.mean(np.log(P[np.arange(len(P)), yd[1300:]])):.4f}")

# ------------------------------------------------------------------ part 6
section("10. Your XGBoost against the real one (18.5.5.2)")
sample = np.random.default_rng(0).permutation(len(X_train))[:1500]
Xs, ys = X_train[sample], y_train[sample]
try:
    import xgboost as xgb
    params = {"objective": "binary:logistic", "tree_method": "exact", "max_depth": 3, "eta": 0.1, "lambda": 1.0, "gamma": 0.0,
              "min_child_weight": 1.0, "base_score": 0.5, "nthread": 1}
    D = xgb.DMatrix(Xs, label=ys)
    real = xgb.train(params, D, num_boost_round=5).predict(xgb.DMatrix(Xs), output_margin=True)
    mine = additive_predict(xgb_fit(Xs, ys, M=5, max_depth=3, max_bins=100000), Xs)
    print(f"5 rounds, largest margin difference: {np.abs(real - mine).max():.1e}")
    real = xgb.train(dict(params, gamma=5.0), D, num_boost_round=30).predict(xgb.DMatrix(Xs), output_margin=True)
    for ours_gamma in (5.0, 2.5):
        mine = additive_predict(xgb_fit(Xs, ys, M=30, max_depth=3, gamma=ours_gamma, max_bins=100000), Xs)
        print(f"30 rounds, library gamma 5 vs book gamma {ours_gamma}: largest margin difference {np.abs(real - mine).max():.1e}")
except ImportError:
    print("xgboost is not installed here, so this comparison is skipped")

section("11. What is second-order information worth? (18.5.5.2)")
for name, r in order_study(X_train, y_train, X_test, y_test, M=200, max_depth=4).items():
    print(f"  {name:<14} best test log loss {r['best_log_loss']:.4f} at {r['best_M']:>3} trees, test error {r['test_error']:.4f}, train log loss {r['train_log_loss']:.4f}")

section("12. Early stopping, lambda and gamma (18.5, eq 18.47)")
perm = np.random.default_rng(1).permutation(len(X_train))
stop = xgb_fit(X_train[perm[766:]], y_train[perm[766:]], M=2000, max_depth=4, eval_set=(X_train[perm[:766]], y_train[perm[:766]]), patience=50)
print(f"patience 50 stopped after {len(stop['val_loss'])} rounds and kept {stop['best_M']}: test log loss {log_loss_of(additive_predict(stop, X_test), y_test):.4f}, error {error_of(additive_predict(stop, X_test), y_test):.4f}")
for lam, gamma in ((0.0, 0.0), (1.0, 0.0), (100.0, 0.0), (1.0, 1.0), (1.0, 5.0)):
    m = xgb_fit(X_train, y_train, M=200, max_depth=4, lam=lam, gamma=gamma)
    F = additive_predict(m, X_test)
    print(f"  lambda {lam:<5} gamma {gamma:<3}  mean leaves {np.mean([(t['feature'] < 0).sum() for t in m['trees']]):5.2f}  test log loss {log_loss_of(F, y_test):.4f}  error {error_of(F, y_test):.4f}")

section("13. Does subsampling generalize better? (18.5.5)")
for label, data, depth, M in (("spam, depth 4", (X_train, y_train, X_test, y_test), 4, 150), ("nested spheres, depth 2", (Xh, yh, Xv, yv), 2, 200)):
    r = subsample_study(*data, [1.0, 0.5], seeds=[0, 1], M=M, max_depth=depth)
    print(f"  {label:<24} full: error {r[1.0]['mean_error']:.4f}, log loss {r[1.0]['mean_log_loss']:.4f}   half: error {r[0.5]['mean_error']:.4f}, log loss {r[0.5]['mean_log_loss']:.4f}")

# ------------------------------------------------------------------ part 7
section("14. Figure 18.5: bagging, random forest, boosting on the fixed spam split (18.4, 18.5)")
curves = ensemble_error_curves(X_train, y_train, X_test, y_test, M_forest=100, M_boost=300, max_depth=4)
for name, curve in curves.items():
    print(f"  {name:<14}", "  ".join(f"{m}: {curve[m - 1]:.4f}" for m in (10, 50, 100, 300) if m <= len(curve)))
rf = forest_fit(X_train, y_train, M=100, max_features=7)
print(f"random forest out-of-bag error {rf['oob_error']:.4f} against test error {staged_error(rf, X_test, y_test, average=True, threshold=0.5)[-1]:.4f}")

section("15. Is the forest's lead because many features are irrelevant? (18.4)")
rng = np.random.default_rng(7)
noise_train, noise_test = rng.standard_normal((len(X_train), 200)), rng.standard_normal((len(X_test), 200))
for label, a, c in (("all 57 features", X_train, X_test), ("57 + 200 pure noise", np.c_[X_train, noise_train], np.c_[X_test, noise_test])):
    p = a.shape[1]
    bag = staged_error(forest_fit(a, y_train, M=30), c, y_test, average=True, threshold=0.5)[-1]
    forest = staged_error(forest_fit(a, y_train, M=30, max_features=int(np.sqrt(p))), c, y_test, average=True, threshold=0.5)[-1]
    print(f"  {label:<20} bagging {bag:.4f}  random forest {forest:.4f}  lead {bag - forest:+.4f}")

section("16. Bias and variance on a known function (18.3, 18.5)")
Xb, _, fb = friedman(300, 9, noise=0.0)
fitters = {
    "one stump": lambda a, b, c: additive_predict(l2boost_fit(a, b, M=1, nu=1.0, max_depth=1), c),
    "bagged stumps": lambda a, b, c: additive_predict(forest_fit(a, b, M=25, max_depth=1), c),
    "boosted stumps": lambda a, b, c: additive_predict(gradient_boost_fit(a, b, M=150, nu=0.3, max_depth=1), c),
    "one deep tree": lambda a, b, c: additive_predict(l2boost_fit(a, b, M=1, nu=1.0, max_depth=12), c),
    "bagged deep trees": lambda a, b, c: additive_predict(forest_fit(a, b, M=25, max_depth=12), c),
    "boosted depth 3": lambda a, b, c: additive_predict(gradient_boost_fit(a, b, M=150, nu=0.1, max_depth=3), c),
}
study = bias_variance_study(lambda seed: friedman(300, 1000 + seed)[:2], Xb, fb, fitters, n_draws=8)
for name, r in study.items():
    print(f"  {name:<18} bias^2 {r['bias2']:7.3f}   variance {r['variance']:7.3f}")

# ------------------------------------------------------------------ part 8
section("17. Figure 18.9: importance on the spam filter (18.6.1)")
boost = gradient_boost_fit(X_train, y_train, loss="logloss", M=300, max_depth=4)
importance = feature_importance(boost["trees"], 57)
order = np.argsort(-importance)
print("  ", ", ".join(f"{SPAM_NAMES[i]} {importance[i]:.0f}" for i in order[:10]))
print(f"   george ranks {int(np.where(order == SPAM_NAMES.index('george'))[0][0]) + 1}, hp ranks {int(np.where(order == SPAM_NAMES.index('hp'))[0][0]) + 1} (the book's figure puts george first and hp second)")

section("18. Figure 18.10: partial dependence in log-odds (18.6.2)")
predict = lambda Z: additive_predict(boost, Z)
background = X_train[np.random.default_rng(0).permutation(len(X_train))[:500]]
for name in ("ch!", "remove", "edu", "hp"):
    k = SPAM_NAMES.index(name)
    grid = np.unique(np.round(np.quantile(X_train[:, k], [0.0, 0.5, 0.9, 0.98]), 3))
    values = partial_dependence(predict, background, k, grid)
    print(f"  {name:<7}", "  ".join(f"{a:g}: {b:+.2f}" for a, b in zip(grid, values)))
hp_grid, bang_grid = np.array([0.0, 1.0, 3.0]), np.array([0.0, 0.3, 1.0])
table = partial_dependence_2d(predict, background, SPAM_NAMES.index("hp"), SPAM_NAMES.index("ch!"), hp_grid, bang_grid)
for a, row in zip(hp_grid, table):
    print(f"  hp = {a:g}: ch! 0 -> 1 moves the log-odds by {row[-1] - row[0]:+.2f} and the probability from {1 / (1 + np.exp(-row[0])):.3f} to {1 / (1 + np.exp(-row[-1])):.3f}")
print(f"\ntotal time {time.time() - START:.0f}s")

