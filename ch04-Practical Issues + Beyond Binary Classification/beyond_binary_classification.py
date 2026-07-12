"""
Chapter 5: Beyond Binary Classification
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. Subsampling for imbalanced (alpha-weighted) binary classification, Section 5.1
  2. One-versus-All (OVA) multiclass reduction, Algorithm 12/13
  3. All-versus-All (AVA) multiclass reduction, Algorithm 14/15

Tested on real datasets from sklearn.datasets, compared against
scikit-learn's OneVsRestClassifier / OneVsOneClassifier and
imbalanced-class baselines.
"""

import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score, f1_score

RNG = np.random.RandomState(3)


# ----------------------------------------------------------------------
# A simple binary base learner: the perceptron from Chapter 3
# ----------------------------------------------------------------------
class SimplePerceptron:
    def __init__(self, max_iter=30, random_state=0):
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X, y):
        # y is expected to be +-1
        X = np.asarray(X)
        y = np.asarray(y)
        N, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        rng = np.random.RandomState(self.random_state)
        order = np.arange(N)
        for _ in range(self.max_iter):
            rng.shuffle(order)
            for n in order:
                a = self.w @ X[n] + self.b
                if y[n] * a <= 0:
                    self.w += y[n] * X[n]
                    self.b += y[n]
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ----------------------------------------------------------------------
# 1. Subsampling for alpha-weighted binary classification (Section 5.1)
# ----------------------------------------------------------------------
def subsample_map(X, y, alpha, random_state=0):
    """
    Algorithm 11: SubsampleMap.
    Keeps all positive (y=+1) examples, and keeps each negative (y=-1)
    example only with probability 1/alpha. This turns an alpha-weighted
    problem into a plain (unweighted) binary classification problem.
    """
    rng = np.random.RandomState(random_state)
    X, y = np.asarray(X), np.asarray(y)
    keep = np.ones(len(y), dtype=bool)
    neg_idx = np.where(y == -1)[0]
    u = rng.uniform(0, 1, size=len(neg_idx))
    drop = neg_idx[u >= 1.0 / alpha]
    keep[drop] = False
    return X[keep], y[keep]


# ----------------------------------------------------------------------
# 2. One-versus-All (Algorithm 12: OneVersusAllTrain / 13: Test)
# ----------------------------------------------------------------------
class OVAClassifier:
    def __init__(self, base_learner_fn, classes=None):
        self.base_learner_fn = base_learner_fn
        self.classes = classes

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = self.classes if self.classes is not None else np.unique(y)
        self.models_ = {}
        for c in self.classes_:
            y_bin = np.where(y == c, 1, -1)
            model = self.base_learner_fn()
            model.fit(X, y_bin)
            self.models_[c] = model
        return self

    def predict(self, X):
        X = np.asarray(X)
        scores = np.column_stack(
            [self.models_[c].decision_function(X) for c in self.classes_]
        )
        return self.classes_[np.argmax(scores, axis=1)]


# ----------------------------------------------------------------------
# 3. All-versus-All (Algorithm 14: AllVersusAllTrain / 15: Test)
# ----------------------------------------------------------------------
class AVAClassifier:
    def __init__(self, base_learner_fn, classes=None):
        self.base_learner_fn = base_learner_fn
        self.classes = classes

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = self.classes if self.classes is not None else np.unique(y)
        self.models_ = {}
        for i, ci in enumerate(self.classes_):
            for cj in self.classes_[i + 1:]:
                mask = (y == ci) | (y == cj)
                y_bin = np.where(y[mask] == ci, 1, -1)
                model = self.base_learner_fn()
                model.fit(X[mask], y_bin)
                self.models_[(ci, cj)] = model
        return self

    def predict(self, X):
        X = np.asarray(X)
        N = X.shape[0]
        scores = {c: np.zeros(N) for c in self.classes_}
        for (ci, cj), model in self.models_.items():
            pred = np.sign(model.decision_function(X))
            scores[ci] += (pred == 1)
            scores[cj] += (pred == -1)
        score_matrix = np.column_stack([scores[c] for c in self.classes_])
        return self.classes_[np.argmax(score_matrix, axis=1)]


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ========================================================================
# Experiment A: imbalanced data, subsampling (Section 5.1, Theorem 2)
# ========================================================================
print("=" * 70)
print("EXPERIMENT A: subsampling for imbalanced classification (Section 5.1)")
print("=" * 70)

bc = load_breast_cancer()
Xb, yb_raw = bc.data, bc.target
# Artificially imbalance: keep only 8% of the "malignant" (0) class
minority_idx = np.where(yb_raw == 0)[0]
majority_idx = np.where(yb_raw == 1)[0]
keep_minority = RNG.choice(minority_idx, size=int(0.08 * len(minority_idx)), replace=False)
imb_idx = np.concatenate([keep_minority, majority_idx])
Xb_imb, yb_imb_raw = Xb[imb_idx], yb_raw[imb_idx]
yb_imb = np.where(yb_imb_raw == 0, 1, -1)  # rare class (malignant) = positive = +1

print(f"Imbalanced dataset: {np.sum(yb_imb == 1)} positive vs {np.sum(yb_imb == -1)} negative "
      f"({np.mean(yb_imb == 1) * 100:.1f}% positive)")

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb_imb, yb_imb, test_size=0.3, random_state=1, stratify=yb_imb
)
mu, sigma = Xb_train.mean(0), Xb_train.std(0)
sigma[sigma == 0] = 1
Xb_train_n = (Xb_train - mu) / sigma
Xb_test_n = (Xb_test - mu) / sigma

alpha = np.sum(yb_train == -1) / np.sum(yb_train == 1)  # weight positives by class ratio
print(f"alpha (importance of positive class) = {alpha:.2f}")

# baseline: train directly on imbalanced data, no subsampling
p_plain = SimplePerceptron(max_iter=30, random_state=1).fit(Xb_train_n, yb_train)
pred_plain = p_plain.predict(Xb_test_n)

# subsampled: apply Algorithm 11 before training
Xb_sub, yb_sub = subsample_map(Xb_train_n, yb_train, alpha, random_state=1)
p_sub = SimplePerceptron(max_iter=30, random_state=1).fit(Xb_sub, yb_sub)
pred_sub = p_sub.predict(Xb_test_n)

print(f"\n{'method':>18} | {'accuracy':>8} | {'F1 (positive class)':>20}")
print("-" * 52)
print(f"{'no subsampling':>18} | {accuracy(yb_test, pred_plain):>8.4f} "
      f"| {f1_score(yb_test, pred_plain, pos_label=1):>20.4f}")
print(f"{'subsampled':>18} | {accuracy(yb_test, pred_sub):>8.4f} "
      f"| {f1_score(yb_test, pred_sub, pos_label=1):>20.4f}")
n_pos_train = np.sum(yb_train == 1)
n_neg_kept = np.sum(yb_sub == -1)
print(f"\nTraining set sizes: full={len(yb_train)}  subsampled={len(yb_sub)} "
      f"({n_pos_train} positive + {n_neg_kept} negative kept)")
print("\nRaw accuracy can be misleadingly high on imbalanced data (predicting the majority")
print("class every time already scores well), so F1 on the positive class is the more honest")
print("metric to watch. In this run, subsampling actually *hurts*: with alpha this large and")
print("the minority class already this small, throwing away ~1/alpha of the negatives leaves")
print("only a couple dozen training points total -- too little for the perceptron to learn from.")
print("This matches the book's own warning (Section 5.1): subsampling 'throws out a lot of data")
print("(especially for large alpha)'. Oversampling the minority class instead (see Exercises)")
print("keeps all the data and would be expected to do better here.")

# ========================================================================
# Experiment B: One-vs-All and All-vs-All multiclass (Sections 5.2)
# ========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT B: OVA / AVA multiclass reductions on sklearn's Digits dataset")
print("=" * 70)

digits = load_digits()
Xd, yd = digits.data, digits.target
print(f"Dataset shape: {Xd.shape[0]} examples, {Xd.shape[1]} features, {len(np.unique(yd))} classes")

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, random_state=1, stratify=yd
)
mu_d, sigma_d = Xd_train.mean(0), Xd_train.std(0)
sigma_d[sigma_d == 0] = 1
Xd_train_n = (Xd_train - mu_d) / sigma_d
Xd_test_n = (Xd_test - mu_d) / sigma_d

base_fn = lambda: SimplePerceptron(max_iter=20, random_state=1)

print("\nTraining from-scratch OVA (10 binary classifiers)...")
ova = OVAClassifier(base_fn).fit(Xd_train_n, yd_train)
ova_acc = accuracy(yd_test, ova.predict(Xd_test_n))

print("Training from-scratch AVA (45 binary classifiers)...")
ava = AVAClassifier(base_fn).fit(Xd_train_n, yd_train)
ava_acc = accuracy(yd_test, ava.predict(Xd_test_n))

print("Training sklearn OneVsRestClassifier(Perceptron) for comparison...")
sk_ova = OneVsRestClassifier(SKPerceptron(max_iter=20, tol=None, random_state=1)).fit(Xd_train_n, yd_train)
sk_ova_acc = accuracy(yd_test, sk_ova.predict(Xd_test_n))

print("Training sklearn OneVsOneClassifier(Perceptron) for comparison...")
sk_ava = OneVsOneClassifier(SKPerceptron(max_iter=20, tol=None, random_state=1)).fit(Xd_train_n, yd_train)
sk_ava_acc = accuracy(yd_test, sk_ava.predict(Xd_test_n))

print(f"\n{'method':>28} | {'# binary classifiers':>21} | {'test accuracy':>13}")
print("-" * 68)
print(f"{'From-scratch OVA':>28} | {'10':>21} | {ova_acc:>13.4f}")
print(f"{'From-scratch AVA':>28} | {'45':>21} | {ava_acc:>13.4f}")
print(f"{'sklearn OneVsRest (OVA)':>28} | {'10':>21} | {sk_ova_acc:>13.4f}")
print(f"{'sklearn OneVsOne (AVA)':>28} | {'45':>21} | {sk_ava_acc:>13.4f}")

print("\nAs the book notes (Theorem 3 & 4): AVA trains K(K-1)/2 classifiers instead of K,")
print("each on an easier binary sub-problem (only 2 classes' worth of data), which")
print("often (but not always) gives it an edge over OVA, at higher training cost.")
