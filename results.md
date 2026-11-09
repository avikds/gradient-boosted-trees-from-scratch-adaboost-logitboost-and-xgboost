```
spam data: UCI copy with a random split of the same size (ESL mirrors unavailable)
3065 training and 1536 test emails, 39.5% spam in training

=== 1. The tree: CART when g = -y and h = 1, and growing before pruning (18.1, 18.5.5.2) [37s]
depth-5 tree vs scikit-learn: largest prediction difference 7.1e-15
XOR, gamma = 2: stop early 1 leaf, grow then prune 4 leaves

=== 2. Least-squares boosting: Figure 18.6 and what shrinkage buys (18.5.2) [37s]
Fig 18.6 data, depth-2 trees, MSE against the noiseless curve: f0 0.05033, f1 0.01139, f2 0.00554, f3 0.00280
  nu = 1.0   best at  14 trees: test MSE 5.463, after 600: 6.197
  nu = 0.3   best at  60 trees: test MSE 2.165, after 600: 2.259
  nu = 0.1   best at 185 trees: test MSE 1.657, after 600: 1.716
  nu = 0.03  best at 600 trees: test MSE 1.778, after 600: 1.778

=== 3. AdaBoost: a stump near chance becomes a strong classifier (18.5, 18.5.3) [88s]
nested spheres, eq 18.29 stumps, test error: 1 rounds 0.4456, 10 rounds 0.4056, 100 rounds 0.2764, 400 rounds 0.2525
training bound: prod 2 sqrt(err (1-err)) = 0.715762, mean exp loss = 0.715762

=== 4. The stumps are the problem, not the loss (18.5.3, 18.5.4) [88s]
test error after 400 stumps on the same data:
  +1/-1 leaves, stump minimizes weighted error (eq 18.29)   0.2525
  +1/-1 leaves, stump minimizes weighted Gini (scikit-learn) 0.1205
  real leaves, exponential loss                             0.0614
  real leaves, log loss (LogitBoost)                        0.0617

=== 5. Can exponential loss give probabilities? The book says no; eq 18.25 says yes (18.5.3, 18.5.4) [93s]
spam, 400 stumps, probabilities from eq 18.36:
  AdaBoost           log loss 0.1766  ECE 0.0177  error 0.0618
  LogitBoost         log loss 0.2985  ECE 0.0398  error 0.0540
  LogitBoost nu=0.1  log loss 0.1768  ECE 0.0155  error 0.0586

=== 6. Mislabeled training data: the mechanism and the consequence (18.5.4) [102s]
  flip 0.0   error AdaBoost 0.0553  LogitBoost 0.0521   weight on flipped: AdaBoost 0.000  LogitBoost 0.000
  flip 0.05  error AdaBoost 0.0671  LogitBoost 0.0592   weight on flipped: AdaBoost 0.284  LogitBoost 0.117
  flip 0.2   error AdaBoost 0.0951  LogitBoost 0.0990   weight on flipped: AdaBoost 0.412  LogitBoost 0.260
same trees (depth 3, nu = 0.1, 150 rounds), only the loss differs, clean test error:
  flip 0.0   exponential 0.0521  log loss 0.0482
  flip 0.3   exponential 0.1445  log loss 0.1276

=== 7. Table 18.1 against finite differences (18.5.5) [132s]
exponential row as printed, -y exp(-y f): largest error 43.525; with the sign fixed: 3.0e-09
log-loss row with pi = sigma(2f): largest error 0.150; with pi = sigma(f): 2.6e-10

=== 8. Leaf re-solve and robust losses (18.5.5, 18.5.5.1) [132s]
  absolute loss, re-solve True : test MAE / scale, y as is 1.255, y * 100 1.255
  absolute loss, re-solve False: test MAE / scale, y as is 1.557, y * 100 3.906
5% of the training targets off by 30, RMSE on clean targets after 300 rounds (best stage in brackets):
  squared          4.706 (3.134)
  absolute         1.767 (1.649)
  huber delta=1.0  3.395 (1.721)
  huber delta=3.0  4.844 (1.893)

=== 9. Softmax boosting on handwritten digits (eq 18.42-18.43) [178s]
    1 rounds: test accuracy 0.8008, log loss 1.7281
   10 rounds: test accuracy 0.8954, log loss 0.6356
  100 rounds: test accuracy 0.9759, log loss 0.0979

=== 10. Your XGBoost against the real one (18.5.5.2) [213s]
5 rounds, largest margin difference: 1.0e-07
30 rounds, library gamma 5 vs book gamma 5.0: largest margin difference 9.1e-01
30 rounds, library gamma 5 vs book gamma 2.5: largest margin difference 6.0e-07

=== 11. What is second-order information worth? (18.5.5.2) [234s]
  first_order    best test log loss 0.2034 at 200 trees, test error 0.0645, train log loss 0.1677
  newton_leaves  best test log loss 0.1323 at 198 trees, test error 0.0469, train log loss 0.0424
  second_order   best test log loss 0.1387 at 195 trees, test error 0.0430, train log loss 0.0477

=== 12. Early stopping, lambda and gamma (18.5, eq 18.47) [287s]
patience 50 stopped after 224 rounds and kept 174: test log loss 0.1457, error 0.0547
  lambda 0.0   gamma 0.0  mean leaves  9.98  test log loss 0.1363  error 0.0488
  lambda 1.0   gamma 0.0  mean leaves  9.59  test log loss 0.1396  error 0.0475
  lambda 100.0 gamma 0.0  mean leaves 11.99  test log loss 0.1637  error 0.0560
  lambda 1.0   gamma 1.0  mean leaves  4.75  test log loss 0.1515  error 0.0534
  lambda 1.0   gamma 5.0  mean leaves  2.54  test log loss 0.1785  error 0.0612

=== 13. Does subsampling generalize better? (18.5.5) [358s]
  spam, depth 4            full: error 0.0456, log loss 0.1354   half: error 0.0472, log loss 0.1386
  nested spheres, depth 2  full: error 0.1095, log loss 0.3524   half: error 0.0936, log loss 0.3127

=== 14. Figure 18.5: bagging, random forest, boosting on the fixed spam split (18.4, 18.5) [410s]
  bagging        10: 0.0684  50: 0.0645  100: 0.0566
  random_forest  10: 0.0612  50: 0.0495  100: 0.0488
  boosting       10: 0.0820  50: 0.0560  100: 0.0475  300: 0.0488
random forest out-of-bag error 0.0483 against test error 0.0488

=== 15. Is the forest's lead because many features are irrelevant? (18.4) [547s]
  all 57 features      bagging 0.0658  random forest 0.0527  lead +0.0130
  57 + 200 pure noise  bagging 0.0749  random forest 0.0742  lead +0.0007

=== 16. Bias and variance on a known function (18.3, 18.5) [864s]
  one stump          bias^2  16.432   variance   1.189
  bagged stumps      bias^2  15.389   variance   0.732
  boosted stumps     bias^2   1.839   variance   0.759
  one deep tree      bias^2   3.579   variance   7.285
  bagged deep trees  bias^2   3.522   variance   1.294
  boosted depth 3    bias^2   1.374   variance   0.970

=== 17. Figure 18.9: importance on the spam filter (18.6.1) [945s]
   ch! 100, ch$ 47, remove 36, hp 27, free 17, your 14, CAPAVE 12, CAPMAX 11, george 10, edu 9
   george ranks 9, hp ranks 4 (the book's figure puts george first and hp second)

=== 18. Figure 18.10: partial dependence in log-odds (18.6.2) [962s]
  ch!     0: -1.89  0.706: -0.29  1.75: +0.85
  remove  0: -1.61  0.4: +0.82  1.47: +1.07
  edu     0: -1.19  0.142: -2.92  2.444: -3.49
  hp      0: -0.90  1.78: -3.75  5.154: -3.62
  hp = 0: ch! 0 -> 1 moves the log-odds by +2.66 and the probability from 0.198 to 0.780
  hp = 1: ch! 0 -> 1 moves the log-odds by +1.57 and the probability from 0.020 to 0.088
  hp = 3: ch! 0 -> 1 moves the log-odds by +1.57 and the probability from 0.020 to 0.091

total time 962s
```
