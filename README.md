# Gradient Boosted Trees from Scratch: AdaBoost, LogitBoost and XGBoost

Build the boosting chapter of Murphy's Probabilistic Machine Learning in NumPy, from one tree that sees only gradients and Hessians. Feed it squared-error statistics and it matches scikit-learn's CART to rounding. Feed it log-loss statistics and your booster matches the real xgboost library margin for margin. On that tree you write least-squares boosting, AdaBoost.M1 derived from exponential loss, LogitBoost, gradient boosting with a leaf re-solve for five losses, softmax boosting for ten digits, and XGBoost's regularized objective with pruning and early stopping. Then you test the chapter on its own data, the 4,601 emails of the spam dataset. The book's spam comparison reproduces: over 11 splits, bagging errs on 5.9% of test emails, a random forest on 5.0%, boosting on 4.8%. Several claims do not survive. Two rows of the book's loss table fail a finite-difference check. AdaBoost's scores turn out to be good probabilities. On one dataset, 400 stumps score 25% or 6% depending on their leaves, not their loss. And partial dependence shows "!" moving the log-odds of spam by about the same amount with or without "hp".

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** fit_bins
- [x] **2.** split_gain
- [x] **3.** gradient_histograms
- [x] **4.** best_split
- [x] **5.** tree_predict
- [x] **6.** grow_tree
- [x] **7.** prune_tree
- [x] **8.** additive_predict
- [x] **9.** l2boost_fit

---

Built on Deep-ML.
