"""
Chapter 4: Practical Issues
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. Feature normalization (centering + variance scaling), Section 4.3
  2. K-fold cross-validation (Algorithm from Section 4.6)
  3. Irrelevant-feature robustness experiment across DT / KNN / Perceptron
  4. Feature pruning by variance, Section 4.3

Tested on real datasets from sklearn.datasets, compared against
scikit-learn's cross_val_score for a sanity check.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import accuracy_score

RNG = np.random.RandomState(7)


# ----------------------------------------------------------------------
# 1. Feature normalization (Section 4.3, Eq 4.1-4.6)
# ----------------------------------------------------------------------
def center_and_scale(X_train, X_test):
    """Center to zero mean and scale to unit variance, fit on train only."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0  # avoid divide-by-zero on constant features
    X_train_norm = (X_train - mu) / sigma
    X_test_norm = (X_test - mu) / sigma
    return X_train_norm, X_test_norm


# ----------------------------------------------------------------------
# 2. K-fold cross-validation (Algorithm 8: CrossValidate, Section 4.6)
# ----------------------------------------------------------------------
def cross_validate(model_fn, X, y, K=10):
    """
    model_fn: a callable that returns a *fresh* unfitted model with a
              .fit(X, y) / .predict(X) interface.
    Returns the list of per-fold accuracies (mimics Algorithm 8's inner loop).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    N = X.shape[0]
    indices = np.arange(N)
    RNG.shuffle(indices)  # important: shuffle before splitting into folds

    fold_sizes = np.full(K, N // K, dtype=int)
    fold_sizes[: N % K] += 1
    scores = []
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current: current + fold_size]
        train_idx = np.setdiff1d(indices, test_idx)
        current += fold_size

        model = model_fn()
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], preds))
    return np.array(scores)


# ----------------------------------------------------------------------
# Experiment A: does centering/scaling matter for perceptron / KNN?
# ----------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT A: effect of feature normalization (Section 4.3)")
print("=" * 70)

data = load_breast_cancer()
X, y_raw = data.data, data.target
y_pm = np.where(y_raw == 0, -1, 1)  # +-1 labels for perceptron

X_train, X_test, y_train, y_test, ytr_pm, yte_pm = train_test_split(
    X, y_raw, y_pm, test_size=0.3, random_state=42, stratify=y_raw
)

X_train_norm, X_test_norm = center_and_scale(X_train, X_test)

print(f"{'model':>28} | {'raw features acc':>17} | {'normalized acc':>15}")
print("-" * 68)

knn_raw = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
knn_norm = KNeighborsClassifier(n_neighbors=5).fit(X_train_norm, y_train)
print(f"{'KNN (k=5)':>28} | {accuracy_score(y_test, knn_raw.predict(X_test)):>17.4f} "
      f"| {accuracy_score(y_test, knn_norm.predict(X_test_norm)):>15.4f}")

perc_raw = SKPerceptron(max_iter=50, tol=None, random_state=1).fit(X_train, ytr_pm)
perc_norm = SKPerceptron(max_iter=50, tol=None, random_state=1).fit(X_train_norm, ytr_pm)
print(f"{'Perceptron':>28} | {accuracy_score(yte_pm, perc_raw.predict(X_test)):>17.4f} "
      f"| {accuracy_score(yte_pm, perc_norm.predict(X_test_norm)):>15.4f}")

dt_raw = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_train, y_train)
dt_norm = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_train_norm, y_train)
print(f"{'Decision Tree (depth=4)':>28} | {accuracy_score(y_test, dt_raw.predict(X_test)):>17.4f} "
      f"| {accuracy_score(y_test, dt_norm.predict(X_test_norm)):>15.4f}")

print("\nAs the book notes (Section 4.3): decision trees are invariant to monotonic")
print("per-feature rescaling (splits don't change), while KNN and the perceptron")
print("are sensitive to feature scale because they rely on raw distances/dot products.")

# ----------------------------------------------------------------------
# Experiment B: robustness to irrelevant (noise) features (Section 4.2)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT B: robustness to irrelevant features (Section 4.2, Figure 4.6)")
print("=" * 70)
print(f"{'# noise feats':>13} | {'DT acc':>7} | {'KNN acc':>8} | {'Perceptron acc':>15}")
print("-" * 55)
for n_noise in [0, 10, 30, 60, 120, 240]:
    noise_train = RNG.normal(0, 1, size=(X_train_norm.shape[0], n_noise))
    noise_test = RNG.normal(0, 1, size=(X_test_norm.shape[0], n_noise))
    Xn_train = np.hstack([X_train_norm, noise_train])
    Xn_test = np.hstack([X_test_norm, noise_test])

    dt = DecisionTreeClassifier(max_depth=4, random_state=1).fit(Xn_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=5).fit(Xn_train, y_train)
    perc = SKPerceptron(max_iter=50, tol=None, random_state=1).fit(Xn_train, ytr_pm)

    dt_acc = accuracy_score(y_test, dt.predict(Xn_test))
    knn_acc = accuracy_score(y_test, knn.predict(Xn_test))
    perc_acc = accuracy_score(yte_pm, perc.predict(Xn_test))
    print(f"{n_noise:>13} | {dt_acc:>7.4f} | {knn_acc:>8.4f} | {perc_acc:>15.4f}")

print("\nKNN degrades fastest: irrelevant features dilute the meaningfulness of distance")
print("(Chapter 2's curse of dimensionality). Decision trees are the most robust because")
print("they explicitly select useful features and can mostly ignore noisy ones.")

# ----------------------------------------------------------------------
# Experiment C: K-fold cross-validation, sanity-checked vs sklearn
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT C: from-scratch 10-fold CV vs sklearn.cross_val_score")
print("=" * 70)

my_scores = cross_validate(lambda: DecisionTreeClassifier(max_depth=4, random_state=1), X, y_raw, K=10)
sk_scores = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=1), X, y_raw, cv=10)

print(f"From-scratch 10-fold CV : mean={my_scores.mean():.4f}  std={my_scores.std():.4f}")
print(f"sklearn 10-fold CV      : mean={sk_scores.mean():.4f}  std={sk_scores.std():.4f}")
print("(Folds differ due to different shuffling, so an exact match isn't expected;")
print(" means should be close.)")

# ----------------------------------------------------------------------
# Experiment D: feature pruning by variance (Section 4.3, Figure 4.9)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT D: pruning low-variance features (Section 4.3)")
print("=" * 70)

variances = X_train.var(axis=0)
order = np.argsort(variances)  # lowest variance first

print(f"{'# features pruned':>18} | {'# features kept':>16} | {'DT test acc':>11}")
print("-" * 52)
for n_pruned in [0, 5, 10, 15, 20, 25, 28]:
    keep_idx = order[n_pruned:]  # drop the lowest-variance features
    if len(keep_idx) == 0:
        continue
    dt = DecisionTreeClassifier(max_depth=4, random_state=1).fit(X_train[:, keep_idx], y_train)
    acc = accuracy_score(y_test, dt.predict(X_test[:, keep_idx]))
    print(f"{n_pruned:>18} | {len(keep_idx):>16} | {acc:>11.4f}")

print("\nAs in the book (Figure 4.9): pruning a few low-variance features doesn't hurt,")
print("and can even help slightly, but pruning too aggressively eventually destroys")
print("useful signal and accuracy collapses.")
