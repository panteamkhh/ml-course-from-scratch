"""
Chapter 3: The Perceptron
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. Vanilla Perceptron (Algorithm 5 / 6 in the book)
  2. Averaged Perceptron (Algorithm 7 in the book)

Tested on a real dataset from sklearn.datasets, compared against
scikit-learn's reference Perceptron, and used to demonstrate:
  - the importance of permuting examples every epoch (Section 3.2)
  - overfitting vs MaxIter, i.e. early stopping (Section 3.2)
  - why the averaged perceptron generalizes better than vanilla (Section 3.6)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as SKPerceptron
from sklearn.metrics import accuracy_score

RNG = np.random.RandomState(0)


# ----------------------------------------------------------------------
# 1. Vanilla Perceptron (Algorithm 5: PerceptronTrain / Algorithm 6: Test)
# ----------------------------------------------------------------------
class PerceptronFromScratch:
    def __init__(self, max_iter=50, permute=True, random_state=0):
        self.max_iter = max_iter
        self.permute = permute
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        N, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        rng = np.random.RandomState(self.random_state)

        order = np.arange(N)
        for _ in range(self.max_iter):
            if self.permute:
                rng.shuffle(order)  # re-permute every epoch (Section 3.2)
            for n in order:
                a = self.w @ X[n] + self.b
                if y[n] * a <= 0:  # the "ya <= 0" trick from Algorithm 5
                    self.w += y[n] * X[n]
                    self.b += y[n]
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


# ----------------------------------------------------------------------
# 2. Averaged Perceptron (Algorithm 7: AveragedPerceptronTrain)
# ----------------------------------------------------------------------
class AveragedPerceptronFromScratch:
    def __init__(self, max_iter=50, permute=True, random_state=0):
        self.max_iter = max_iter
        self.permute = permute
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        N, D = X.shape
        w = np.zeros(D)
        b = 0.0
        u = np.zeros(D)   # cached weights
        beta = 0.0        # cached bias
        c = 1             # example counter
        rng = np.random.RandomState(self.random_state)

        order = np.arange(N)
        for _ in range(self.max_iter):
            if self.permute:
                rng.shuffle(order)
            for n in order:
                if y[n] * (w @ X[n] + b) <= 0:
                    w += y[n] * X[n]
                    b += y[n]
                    u += y[n] * c * X[n]
                    beta += y[n] * c
                c += 1

        self.w = w - u / c
        self.b = b - beta / c
        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ========================================================================
# Experiment A: sanity check vs sklearn.linear_model.Perceptron
# ========================================================================
print("=" * 70)
print("EXPERIMENT A: Perceptron on sklearn's Breast Cancer Wisconsin dataset")
print("=" * 70)

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)  # perceptron wants +-1 labels
print(f"Dataset shape: {X.shape[0]} examples, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

my_perc = PerceptronFromScratch(max_iter=20, permute=True, random_state=1).fit(X_train_s, y_train)
my_pred = my_perc.predict(X_test_s)
my_acc = accuracy_score(y_test, my_pred)

sk_perc = SKPerceptron(max_iter=20, tol=None, random_state=1).fit(X_train_s, y_train)
sk_pred = sk_perc.predict(X_test_s)
sk_acc = accuracy_score(y_test, sk_pred)

print(f"From-scratch Perceptron test accuracy : {my_acc:.4f}")
print(f"sklearn Perceptron test accuracy      : {sk_acc:.4f}")
print("(Exact match is not expected: both are order-sensitive, non-convex-optimization")
print(" algorithms with different internal tie-breaking / update schedules.)")

# ========================================================================
# Experiment B: does the data need to be linearly separable to converge?
# ========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT B: permuting data each epoch vs. fixed order (Section 3.2)")
print("=" * 70)
print(f"{'epochs':>7} | {'no permute (train acc)':>24} | {'permute each epoch (train acc)':>32}")
print("-" * 68)
for epochs in [1, 2, 5, 10, 20]:
    p_fixed = PerceptronFromScratch(max_iter=epochs, permute=False, random_state=1).fit(X_train_s, y_train)
    p_perm = PerceptronFromScratch(max_iter=epochs, permute=True, random_state=1).fit(X_train_s, y_train)
    acc_fixed = accuracy(y_train, p_fixed.predict(X_train_s))
    acc_perm = accuracy(y_train, p_perm.predict(X_train_s))
    print(f"{epochs:>7} | {acc_fixed:>24.4f} | {acc_perm:>32.4f}")

# ========================================================================
# Experiment C: overfitting vs MaxIter (need for early stopping)
# ========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT C: train/test accuracy vs MaxIter (Section 3.2, Figure 3.3)")
print("=" * 70)
print(f"{'MaxIter':>8} | {'train acc':>10} | {'test acc':>9}")
print("-" * 34)
for it in [1, 2, 5, 10, 20, 50, 100, 200]:
    p = PerceptronFromScratch(max_iter=it, permute=True, random_state=1).fit(X_train_s, y_train)
    tr_acc = accuracy(y_train, p.predict(X_train_s))
    te_acc = accuracy(y_test, p.predict(X_test_s))
    print(f"{it:>8} | {tr_acc:>10.4f} | {te_acc:>9.4f}")

# ========================================================================
# Experiment D: vanilla vs averaged perceptron generalization (Section 3.6)
# ========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT D: vanilla vs. averaged perceptron (Section 3.6)")
print("=" * 70)
print(f"{'MaxIter':>8} | {'vanilla test acc':>17} | {'averaged test acc':>18}")
print("-" * 48)
for it in [1, 5, 20, 50, 100, 200]:
    p_van = PerceptronFromScratch(max_iter=it, permute=True, random_state=1).fit(X_train_s, y_train)
    p_avg = AveragedPerceptronFromScratch(max_iter=it, permute=True, random_state=1).fit(X_train_s, y_train)
    acc_van = accuracy(y_test, p_van.predict(X_test_s))
    acc_avg = accuracy(y_test, p_avg.predict(X_test_s))
    print(f"{it:>8} | {acc_van:>17.4f} | {acc_avg:>18.4f}")

print("\nAs the book predicts (Section 3.6): the averaged perceptron is more stable")
print("and tends to generalize at least as well as, and often better than, vanilla,")
print("especially once the number of epochs grows large enough that vanilla starts overfitting.")
