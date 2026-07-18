"""
Chapter 10: Ensemble Methods + Efficient Learning
A Course in Machine Learning (Hal Daume III), Chapters 11 & 12

From-scratch implementations of:
  1. Decision Stump (weak learner for boosting)
  2. AdaBoost (Algorithm 11.2 in the book)
  3. Bagging (bootstrap aggregation)
  4. Random Forest (random feature subsets, Algorithm 11.3)
  5. Stochastic Gradient Descent for a regularized linear classifier
     (Chapter 12), compared against full-batch gradient descent
  6. Feature hashing (Section 12.4)

Tested on real datasets from sklearn.datasets and compared against
scikit-learn's reference implementations.
"""

import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score

RNG = np.random.RandomState(0)


# ==========================================================================
# 1. Decision Stump — a decision tree of depth 1 (Section 11.2 of the book)
# ==========================================================================
class DecisionStump:
    """A weak learner: picks a single (feature, threshold, sign) that
    minimizes *weighted* classification error. Labels are assumed to be
    in {-1, +1}."""

    def fit(self, X, y, sample_weight):
        N, D = X.shape
        best_err = np.inf
        best = None
        # Try a handful of candidate thresholds per feature (percentiles)
        # instead of every single value, for speed.
        for d in range(D):
            col = X[:, d]
            thresholds = np.percentile(col, np.linspace(5, 95, 19))
            for thresh in thresholds:
                for polarity in (1, -1):
                    pred = np.where(polarity * (col - thresh) >= 0, 1, -1)
                    err = np.sum(sample_weight[pred != y])
                    if err < best_err:
                        best_err = err
                        best = (d, thresh, polarity)
        self.feature, self.threshold, self.polarity = best
        self.train_err_ = best_err
        return self

    def predict(self, X):
        col = X[:, self.feature]
        return np.where(self.polarity * (col - self.threshold) >= 0, 1, -1)


# ==========================================================================
# 2. AdaBoost (Algorithm 11.2 in the book, "AdaBoost")
# ==========================================================================
class AdaBoostFromScratch:
    def __init__(self, n_rounds=50):
        self.n_rounds = n_rounds

    def fit(self, X, y):
        N = X.shape[0]
        d = np.full(N, 1.0 / N)          # d^(0): uniform importance
        self.stumps = []
        self.alphas = []

        for k in range(self.n_rounds):
            stump = DecisionStump().fit(X, y, d)
            pred = stump.predict(X)
            eps = np.sum(d[pred != y])
            eps = np.clip(eps, 1e-10, 1 - 1e-10)  # avoid log(0)/div by 0
            alpha = 0.5 * np.log((1 - eps) / eps)

            d = d * np.exp(-alpha * y * pred)
            d = d / d.sum()                        # renormalize

            self.stumps.append(stump)
            self.alphas.append(alpha)

            if eps >= 0.5:
                # weak learning assumption violated; stop early
                break
        return self

    def decision_function(self, X):
        agg = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas, self.stumps):
            agg += alpha * stump.predict(X)
        return agg

    def predict(self, X):
        return np.sign(self.decision_function(X)).astype(int)


# ==========================================================================
# 3. Bagging (bootstrap aggregation, Section 11.1)
# ==========================================================================
class BaggingFromScratch:
    def __init__(self, n_estimators=25, max_depth=None, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        N = X.shape[0]
        self.trees = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, N, size=N)      # sample WITH replacement
            Xb, yb = X[idx], y[idx]
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                           random_state=rng.randint(1e9))
            tree.fit(Xb, yb)
            self.trees.append(tree)
        return self

    def predict(self, X):
        # majority vote across bootstrapped trees
        votes = np.array([t.predict(X) for t in self.trees])
        # votes shape: (n_estimators, N); take per-column majority
        preds = np.empty(X.shape[0], dtype=votes.dtype)
        for i in range(X.shape[0]):
            values, counts = np.unique(votes[:, i], return_counts=True)
            preds[i] = values[np.argmax(counts)]
        return preds


# ==========================================================================
# 4. Random Forest (random feature subsets, Algorithm 11.3)
# ==========================================================================
class RandomForestFromScratch:
    def __init__(self, n_estimators=50, max_depth=5, max_features="sqrt",
                 random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        N, D = X.shape
        if self.max_features == "sqrt":
            m = max(1, int(np.sqrt(D)))
        else:
            m = D

        self.trees = []
        self.feature_subsets = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, N, size=N)          # bootstrap sample
            feat_idx = rng.choice(D, size=m, replace=False)  # random features
            Xb, yb = X[idx][:, feat_idx], y[idx]
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                           random_state=rng.randint(1e9))
            tree.fit(Xb, yb)
            self.trees.append(tree)
            self.feature_subsets.append(feat_idx)
        return self

    def predict(self, X):
        votes = np.array([t.predict(X[:, feat_idx])
                           for t, feat_idx in zip(self.trees, self.feature_subsets)])
        preds = np.empty(X.shape[0], dtype=votes.dtype)
        for i in range(X.shape[0]):
            values, counts = np.unique(votes[:, i], return_counts=True)
            preds[i] = values[np.argmax(counts)]
        return preds


# ==========================================================================
# 5. Stochastic Gradient Descent vs full-batch Gradient Descent (Chapter 12)
#    Regularized linear classifier with logistic loss.
# ==========================================================================
def logistic_loss_grad_batch(w, X, y, lam):
    """Full gradient over the entire dataset."""
    z = y * (X @ w)
    # d/dw of sum(log(1+exp(-z))) = sum(-y*x * sigmoid(-z))
    sig = 1.0 / (1.0 + np.exp(z))
    grad = -(X * (y * sig)[:, None]).sum(axis=0) + lam * w
    return grad


def train_batch_gd(X, y, lam=0.01, iters=200, lr=0.1):
    N, D = X.shape
    w = np.zeros(D)
    for k in range(1, iters + 1):
        g = logistic_loss_grad_batch(w, X, y, lam) / N
        w -= lr * g
    return w


def train_sgd(X, y, lam=0.01, epochs=5, lr0=1.0, batch_size=1, seed=0):
    rng = np.random.RandomState(seed)
    N, D = X.shape
    w = np.zeros(D)
    step = 0
    for epoch in range(epochs):
        perm = rng.permutation(N)          # re-permute every epoch (Ch. 12)
        for start in range(0, N, batch_size):
            step += 1
            batch = perm[start:start + batch_size]
            Xb, yb = X[batch], y[batch]
            g = logistic_loss_grad_batch(w, Xb, yb, lam * batch_size / N) / len(batch)
            eta = lr0 / np.sqrt(step)      # decaying learning rate (Ch. 12)
            w -= eta * g
    return w


def logistic_predict(w, X):
    return np.where(X @ w >= 0, 1, -1)


# ==========================================================================
# 6. Feature Hashing (Section 12.4)
# ==========================================================================
def hash_features(X, P, seed=0):
    """Collapse D-dimensional dense features into P-dimensional hashed
    features, following Section 12.4's hashing map phi: R^D -> R^P."""
    N, D = X.shape
    rng = np.random.RandomState(seed)
    hash_idx = rng.randint(0, P, size=D)          # h(d) for each feature d
    Xh = np.zeros((N, P))
    for d in range(D):
        Xh[:, hash_idx[d]] += X[:, d]
    return Xh


# ==========================================================================
# MAIN EXPERIMENTS
# ==========================================================================
if __name__ == "__main__":

    # ----------------------------------------------------------------
    # Data: Breast Cancer Wisconsin, labels remapped to {-1, +1}
    # ----------------------------------------------------------------
    data = load_breast_cancer()
    X, y_raw = data.data, data.target
    y = np.where(y_raw == 0, -1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("=" * 72)
    print("EXPERIMENT A: AdaBoost from scratch vs sklearn AdaBoostClassifier")
    print("=" * 72)
    my_ada = AdaBoostFromScratch(n_rounds=50).fit(X_train_s, y_train)
    my_pred = my_ada.predict(X_test_s)
    my_acc = accuracy_score(y_test, my_pred)

    sk_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50, random_state=42
    ).fit(X_train_s, y_train)
    sk_pred = sk_ada.predict(X_test_s)
    sk_acc = accuracy_score(y_test, sk_pred)

    print(f"From-scratch AdaBoost (50 stumps) test accuracy : {my_acc:.4f}")
    print(f"sklearn AdaBoostClassifier      test accuracy : {sk_acc:.4f}")
    print(f"Number of stumps actually used   : {len(my_ada.stumps)}")

    print("\nTrain error of each individual stump (first 10 rounds):")
    for i, s in enumerate(my_ada.stumps[:10]):
        print(f"  round {i+1:2d}: weighted train err = {s.train_err_:.4f}, alpha = {my_ada.alphas[i]:.4f}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT B: Bagging from scratch vs sklearn BaggingClassifier")
    print("=" * 72)
    my_bag = BaggingFromScratch(n_estimators=25, max_depth=None, random_state=1).fit(
        X_train_s, y_train)
    my_bag_acc = accuracy_score(y_test, my_bag.predict(X_test_s))

    single_tree = DecisionTreeClassifier(random_state=1).fit(X_train_s, y_train)
    single_tree_acc = accuracy_score(y_test, single_tree.predict(X_test_s))

    sk_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(), n_estimators=25, random_state=1
    ).fit(X_train_s, y_train)
    sk_bag_acc = accuracy_score(y_test, sk_bag.predict(X_test_s))

    print(f"Single (unbagged) decision tree test accuracy : {single_tree_acc:.4f}")
    print(f"From-scratch Bagging (25 trees) test accuracy : {my_bag_acc:.4f}")
    print(f"sklearn BaggingClassifier       test accuracy : {sk_bag_acc:.4f}")
    print("(Bagging reduces variance vs a single fully-grown, overfit tree.)")

    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT C: Random Forest from scratch vs sklearn RandomForestClassifier")
    print("=" * 72)
    my_rf = RandomForestFromScratch(n_estimators=100, max_depth=5, random_state=2).fit(
        X_train_s, y_train)
    my_rf_acc = accuracy_score(y_test, my_rf.predict(X_test_s))

    sk_rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, max_features="sqrt", random_state=2
    ).fit(X_train_s, y_train)
    sk_rf_acc = accuracy_score(y_test, sk_rf.predict(X_test_s))

    print(f"From-scratch Random Forest test accuracy : {my_rf_acc:.4f}")
    print(f"sklearn RandomForestClassifier test accuracy : {sk_rf_acc:.4f}")

    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT D: Boosting depth vs #rounds (shallow trees + boosting)")
    print("=" * 72)
    print(f"{'#rounds':>8} | {'test acc':>9}")
    for n_rounds in [1, 2, 5, 10, 25, 50, 100]:
        ada = AdaBoostFromScratch(n_rounds=n_rounds).fit(X_train_s, y_train)
        acc = accuracy_score(y_test, ada.predict(X_test_s))
        print(f"{n_rounds:>8} | {acc:>9.4f}")
    print("(Even depth-1 'stumps' become a strong classifier once boosted.)")

    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT E: Stochastic Gradient Descent vs full-batch Gradient Descent")
    print("=" * 72)

    t0 = time.time()
    w_batch = train_batch_gd(X_train_s, y_train, lam=0.01, iters=200, lr=0.5)
    t_batch = time.time() - t0
    acc_batch = accuracy_score(y_test, logistic_predict(w_batch, X_test_s))

    t0 = time.time()
    w_sgd = train_sgd(X_train_s, y_train, lam=0.01, epochs=10, lr0=1.0, batch_size=1)
    t_sgd = time.time() - t0
    acc_sgd = accuracy_score(y_test, logistic_predict(w_sgd, X_test_s))

    print(f"Full-batch GD  (200 full-dataset sweeps): acc={acc_batch:.4f}, time={t_batch*1000:.2f} ms")
    print(f"Stochastic GD  (10 epochs, batch size 1): acc={acc_sgd:.4f}, time={t_sgd*1000:.2f} ms")
    print("(On this small dataset, batch GD's vectorized numpy ops beat SGD's")
    print(" per-example Python loop in wall-clock time. But SGD reaches similar")
    print(" accuracy after seeing only 10*N examples total instead of 200*N, and")
    print(" each individual update is O(D) instead of O(ND) -- on datasets with")
    print(" millions of rows, that difference is what makes SGD the only option.)")

    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT F: Feature Hashing (Section 12.4) — memory vs accuracy")
    print("=" * 72)
    D = X_train_s.shape[1]
    print(f"Original dimensionality D = {D}")
    print(f"{'P (hashed dim)':>15} | {'test acc':>9}")
    for P in [5, 10, 20, 30, D]:
        Xh_train = hash_features(X_train_s, P, seed=0)
        Xh_test = hash_features(X_test_s, P, seed=0)
        w_hash = train_batch_gd(Xh_train, y_train, lam=0.01, iters=200, lr=0.5)
        acc_hash = accuracy_score(y_test, logistic_predict(w_hash, Xh_test))
        print(f"{P:>15} | {acc_hash:>9.4f}")
    print("(Accuracy degrades gracefully as P shrinks below D, illustrating")
    print(" the collision/variance trade-off described in Section 12.4.)")
