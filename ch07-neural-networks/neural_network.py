"""
Chapter 7: Neural Networks
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. A two-layer neural network (Algorithm 8.2: TwoLayerNetworkTrain,
     back-propagation = gradient descent + chain rule)
  2. Prediction (Algorithm 8.1: TwoLayerNetworkPredict)

Tested on a real dataset (sklearn's Breast Cancer Wisconsin), compared
against scikit-learn's MLPClassifier, and used to demonstrate the
underfitting/overfitting trade-off controlled by the number of hidden
units (Section 8.1, "how many hidden units should I have?").
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

RNG = np.random.RandomState(0)


# ----------------------------------------------------------------------
# Two-layer neural network, implemented from scratch (Section 8.1 - 8.2)
# ----------------------------------------------------------------------
class TwoLayerNetFromScratch:
    """
    Architecture:  D inputs -> K hidden units (tanh) -> 1 output (linear)
    Loss:          squared error (Eq 8.5)
    Training:      full-batch gradient descent via back-propagation
    """

    def __init__(self, n_hidden=10, lr=0.05, n_iter=2000, seed=0):
        self.n_hidden = n_hidden
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)  # labels are +-1
        n, d = X.shape
        rng = np.random.RandomState(self.seed)

        # small random initialization (Section 8.3): avoids the trivial
        # symmetric local optimum you get from initializing at zero.
        # NOTE: bias terms are essential (Section 3.1 / Fig 3.1 "EXPLAIN
        # BIAS"): without a bias, tanh(w.x) is an odd function of x, so a
        # network with no biases can only represent functions that are odd
        # in x. XOR's target is *even* in x (flipping the sign of both
        # inputs does not flip the label), so a bias-free network can
        # provably never solve it -- this is why we add b (hidden bias)
        # and c (output bias) below.
        self.W = rng.uniform(-1.0, 1.0, size=(d, self.n_hidden))
        self.b = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        self.v = rng.uniform(-1.0, 1.0, size=self.n_hidden)
        self.c = 0.0

        for _ in range(self.n_iter):
            # ---- forward propagation (Algorithm 8.1) ----
            a = X @ self.W + self.b         # (n, K) pre-activations
            h = np.tanh(a)                  # (n, K) hidden activations
            y_hat = h @ self.v + self.c     # (n,)   output (linear unit)

            # ---- back-propagation (Section 8.2) ----
            e = y - y_hat                    # (n,) error per example
            grad_v = -(e[:, None] * h).sum(axis=0)              # Eq 8.6
            grad_c = -e.sum()
            # d(loss)/d(hidden activation) = -e * v ; times tanh'(a)
            delta = (-e[:, None] * self.v[None, :]) * (1 - h ** 2)  # Eq 8.11
            grad_W = X.T @ delta            # (D, K)
            grad_b = delta.sum(axis=0)

            self.v -= self.lr * grad_v / n
            self.c -= self.lr * grad_c / n
            self.W -= self.lr * grad_W / n
            self.b -= self.lr * grad_b / n

        return self

    def decision_function(self, X):
        X = np.asarray(X)
        h = np.tanh(X @ self.W + self.b)
        return h @ self.v + self.c

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ----------------------------------------------------------------------
# Experiment A: load real data, correctness check vs sklearn MLP
# ----------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT A: Two-layer net on sklearn's Breast Cancer dataset")
print("=" * 70)

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)  # +-1 labels, per book convention
print(f"Dataset shape: {X.shape[0]} examples, {X.shape[1]} features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n--- Correctness check: from-scratch net vs sklearn MLPClassifier ---")
my_net = TwoLayerNetFromScratch(n_hidden=10, lr=0.1, n_iter=3000, seed=1)
my_net.fit(X_train_s, y_train)
my_pred = my_net.predict(X_test_s)
my_acc = accuracy_score(y_test, my_pred)

sk_mlp = MLPClassifier(
    hidden_layer_sizes=(10,), activation="tanh", solver="lbfgs",
    max_iter=3000, random_state=1,
)
sk_mlp.fit(X_train_s, y_train)
sk_pred = sk_mlp.predict(X_test_s)
sk_acc = accuracy_score(y_test, sk_pred)

print(f"From-scratch net accuracy : {my_acc:.4f}")
print(f"sklearn MLP accuracy      : {sk_acc:.4f}")
print("(Different solvers/initializations -> not identical, but both should")
print(" comfortably beat majority-class baseline and land in a similar range.)")

# ----------------------------------------------------------------------
# Experiment B: underfitting/overfitting vs number of hidden units
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT B: train/test accuracy vs hidden units K")
print("=" * 70)
print(f"{'K':>4} | {'train acc':>10} | {'test acc':>9}")
print("-" * 30)
for k in [1, 2, 5, 10, 20, 50, 100]:
    net = TwoLayerNetFromScratch(n_hidden=k, lr=0.05, n_iter=3000, seed=1)
    net.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, net.predict(X_train_s))
    test_acc = accuracy_score(y_test, net.predict(X_test_s))
    print(f"{k:>4} | {train_acc:>10.4f} | {test_acc:>9.4f}")

# ----------------------------------------------------------------------
# Experiment C: underfitting/overfitting vs number of training iterations
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT C: train/test accuracy vs training iterations (K=20)")
print("=" * 70)
print(f"{'iters':>6} | {'train acc':>10} | {'test acc':>9}")
print("-" * 32)
for n_iter in [10, 50, 200, 1000, 3000, 8000]:
    net = TwoLayerNetFromScratch(n_hidden=20, lr=0.1, n_iter=n_iter, seed=1)
    net.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, net.predict(X_train_s))
    test_acc = accuracy_score(y_test, net.predict(X_test_s))
    print(f"{n_iter:>6} | {train_acc:>10.4f} | {test_acc:>9.4f}")

print("\nAs the book notes (Section 8.1 and Figure 3.3-style curves): more")
print("hidden units and more iterations both increase the model's capacity")
print("to fit the training data, but test accuracy eventually plateaus or")
print("degrades once the network starts fitting noise in the training set.")

# ----------------------------------------------------------------------
# Experiment D: XOR problem - the classic proof that 2-layer nets beat
# a single linear unit (perceptron) on non-linear problems (Section 3.7 / 8.1)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT D: XOR problem (perceptron cannot solve; 2-layer net can)")
print("=" * 70)
X_xor = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
y_xor = np.array([-1, 1, 1, -1], dtype=float)

xor_net = TwoLayerNetFromScratch(n_hidden=4, lr=0.5, n_iter=5000, seed=3)
xor_net.fit(X_xor, y_xor)
xor_pred = xor_net.predict(X_xor)
print("Inputs:\n", X_xor)
print("True labels:     ", y_xor)
print("Predicted labels:", xor_pred)
print(f"Accuracy: {accuracy_score(y_xor, xor_pred):.4f}  (a linear perceptron cannot exceed 0.75 here)")
