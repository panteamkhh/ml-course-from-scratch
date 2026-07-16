"""
Chapter 8: Kernel Methods
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. The kernel trick: linear, polynomial and RBF kernels (Section 9.1, 9.4)
  2. Kernelized Perceptron (Algorithm 9.2: KernelizedPerceptronTrain)
  3. One-vs-rest wrapper to use the binary kernel perceptron on a
     multi-class, real-world dataset

Tested on real datasets from sklearn.datasets (Wine, Digits), compared
against scikit-learn's SVC (which is also a kernel machine), and used to
demonstrate that kernels let a linear-style learner solve problems that
are not linearly separable in the original feature space.
"""

import numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

RNG = np.random.RandomState(0)


# ----------------------------------------------------------------------
# Kernels (Section 9.1 and 9.4)
# ----------------------------------------------------------------------
def linear_kernel(X, Z):
    return X @ Z.T


def polynomial_kernel(X, Z, degree=3):
    # K(x, z) = (1 + x . z)^d   (Eq 9.13)
    return (1.0 + X @ Z.T) ** degree


def rbf_kernel(X, Z, gamma=0.05):
    # K(x, z) = exp(-gamma ||x - z||^2)   (Eq 9.18)
    sq_x = np.sum(X ** 2, axis=1)[:, None]
    sq_z = np.sum(Z ** 2, axis=1)[None, :]
    sq_dists = sq_x + sq_z - 2 * X @ Z.T
    sq_dists = np.maximum(sq_dists, 0.0)  # guard against tiny negatives
    return np.exp(-gamma * sq_dists)


KERNELS = {
    "linear": linear_kernel,
    "poly": lambda X, Z: polynomial_kernel(X, Z, degree=3),
    "rbf": lambda X, Z: rbf_kernel(X, Z, gamma=0.05),
}


# ----------------------------------------------------------------------
# Kernelized Perceptron, implemented from scratch (Algorithm 9.2)
# ----------------------------------------------------------------------
class KernelPerceptronFromScratch:
    def __init__(self, kernel="rbf", n_iter=20):
        self.kernel_name = kernel
        self.kernel_fn = KERNELS[kernel]
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)  # +-1 labels
        n = X.shape[0]
        self.X_train = X
        self.alpha = np.zeros(n)
        self.b = 0.0

        K = self.kernel_fn(X, X)  # (n, n) kernel (Gram) matrix, precomputed once

        for _ in range(self.n_iter):
            for i in range(n):
                # activation only ever depends on kernel products (Eq 9.7)
                a = np.sum(self.alpha * y * K[i]) + self.b
                if y[i] * a <= 0:
                    self.alpha[i] += 1  # mistake-driven update (Algorithm 9.2)
                    self.b += y[i]
        # fold y into alpha once, so prediction is a simple dot product
        self._coef = self.alpha * y
        return self

    def decision_function(self, X):
        X = np.asarray(X)
        K = self.kernel_fn(X, self.X_train)  # (n_test, n_train)
        return K @ self._coef + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


class OneVsRestKernelPerceptron:
    """Multi-class wrapper (Section 5.2, OVA) around the binary kernel perceptron."""

    def __init__(self, kernel="rbf", n_iter=20):
        self.kernel = kernel
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        for c in self.classes_:
            y_bin = np.where(y == c, 1, -1)
            self.models_[c] = KernelPerceptronFromScratch(
                kernel=self.kernel, n_iter=self.n_iter
            ).fit(X, y_bin)
        return self

    def predict(self, X):
        scores = np.column_stack(
            [self.models_[c].decision_function(X) for c in self.classes_]
        )
        return self.classes_[np.argmax(scores, axis=1)]


# ----------------------------------------------------------------------
# Experiment A: Wine dataset (3 classes) -- linear vs poly vs RBF kernel
# ----------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT A: Kernelized Perceptron (OVA) on sklearn's Wine dataset")
print("=" * 70)

wine = load_wine()
Xw, yw = wine.data, wine.target
print(f"Dataset shape: {Xw.shape[0]} examples, {Xw.shape[1]} features, {len(set(yw))} classes")

Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.3, random_state=42, stratify=yw
)
scaler = StandardScaler().fit(Xw_train)
Xw_train_s = scaler.transform(Xw_train)
Xw_test_s = scaler.transform(Xw_test)

print(f"\n{'kernel':>8} | {'train acc':>10} | {'test acc':>9}")
print("-" * 34)
for kname in ["linear", "poly", "rbf"]:
    model = OneVsRestKernelPerceptron(kernel=kname, n_iter=20).fit(Xw_train_s, yw_train)
    train_acc = accuracy_score(yw_train, model.predict(Xw_train_s))
    test_acc = accuracy_score(yw_test, model.predict(Xw_test_s))
    print(f"{kname:>8} | {train_acc:>10.4f} | {test_acc:>9.4f}")

print("\n--- Sanity check vs sklearn.svm.SVC (also a kernel machine) ---")
for kname, sk_kernel in [("linear", "linear"), ("poly", "poly"), ("rbf", "rbf")]:
    svc = SVC(kernel=sk_kernel, degree=3, gamma=0.05 if sk_kernel == "rbf" else "scale")
    svc.fit(Xw_train_s, yw_train)
    acc = accuracy_score(yw_test, svc.predict(Xw_test_s))
    print(f"SVC(kernel={sk_kernel:<7}) test acc: {acc:.4f}")

# ----------------------------------------------------------------------
# Experiment B: hand-written digits, 3 vs 8 (visually similar, genuinely
# non-linear boundary) -- shows RBF/poly beating a plain linear kernel
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT B: digit '3' vs '8' -- non-linear boundary needed")
print("=" * 70)

digits = load_digits()
mask = np.isin(digits.target, [3, 8])
Xd, yd_raw = digits.data[mask], digits.target[mask]
yd = np.where(yd_raw == 3, 1, -1)
print(f"Dataset shape: {Xd.shape[0]} examples, {Xd.shape[1]} pixel features (8x8 images)")

Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, random_state=42, stratify=yd
)
scaler_d = StandardScaler().fit(Xd_train)
Xd_train_s = scaler_d.transform(Xd_train)
Xd_test_s = scaler_d.transform(Xd_test)

print(f"\n{'kernel':>8} | {'train acc':>10} | {'test acc':>9}")
print("-" * 34)
for kname in ["linear", "poly", "rbf"]:
    model = KernelPerceptronFromScratch(kernel=kname, n_iter=20).fit(Xd_train_s, yd_train)
    train_acc = accuracy_score(yd_train, model.predict(Xd_train_s))
    test_acc = accuracy_score(yd_test, model.predict(Xd_test_s))
    print(f"{kname:>8} | {train_acc:>10.4f} | {test_acc:>9.4f}")

# ----------------------------------------------------------------------
# Experiment C: support vectors -- confirm the "confusable pairs" story
# from Section 9.6 (only points near the margin get non-zero alpha)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT C: how many training points actually matter? (Section 9.6)")
print("=" * 70)
rbf_model = KernelPerceptronFromScratch(kernel="rbf", n_iter=20).fit(Xd_train_s, yd_train)
n_touched = np.sum(rbf_model.alpha > 0)
print(f"Training examples that ever triggered a perceptron update: "
      f"{n_touched} / {len(yd_train)} ({100 * n_touched / len(yd_train):.1f}%)")
print("These are examples the (kernelized) perceptron found 'confusable' at some point --")
print("conceptually similar to the support vectors of an SVM (Section 9.6).")
