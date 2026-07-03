"""
Chapter 1: Decision Trees
--------------------------
From-scratch implementation of a Decision Tree classifier using
Information Gain (Entropy), following "A Course in Machine Learning"
by Hal Daume III.

Demo dataset: UCI Breast Cancer Wisconsin (real-world, built into sklearn)
Goal: classify tumors as malignant or benign based on cell measurements.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # for comparison only


# ---------------------------------------------------------------------
# 1. Core math: Entropy & Information Gain
# ---------------------------------------------------------------------
def entropy(y):
    """Measures 'impurity' of a label set. 0 = pure, 1 = maximally mixed (binary)."""
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))


def information_gain(y, y_left, y_right):
    """How much does splitting into y_left/y_right reduce entropy vs y?"""
    n = len(y)
    weighted_child_entropy = (len(y_left) / n) * entropy(y_left) + \
                              (len(y_right) / n) * entropy(y_right)
    return entropy(y) - weighted_child_entropy


# ---------------------------------------------------------------------
# 2. Tree node + recursive builder
# ---------------------------------------------------------------------
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature      # which feature index to split on
        self.threshold = threshold  # split point (<= threshold -> left)
        self.left = left
        self.right = right
        self.label = label          # only set for leaf nodes


class DecisionTreeScratch:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build(X, y, depth=0)
        return self

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        n_features = X.shape[1]

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                if left_mask.sum() == 0 or (~left_mask).sum() == 0:
                    continue
                gain = information_gain(y, y[left_mask], y[~left_mask])
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t

        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        # Stopping conditions (this is exactly "underfitting vs overfitting"
        # control from the book -- max_depth is our hyperparameter)
        if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < 2:
            majority_label = np.bincount(y).argmax()
            return Node(label=majority_label)

        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            majority_label = np.bincount(y).argmax()
            return Node(label=majority_label)

        left_mask = X[:, feat] <= thresh
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def _predict_one(self, x, node):
        if node.label is not None:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


# ---------------------------------------------------------------------
# 3. Demo on real data: Breast Cancer Wisconsin dataset
# ---------------------------------------------------------------------
def main():
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = malignant, 1 = benign
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Our from-scratch tree ---
    tree = DecisionTreeScratch(max_depth=4)
    tree.fit(X_train, y_train)
    preds = tree.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"[From-scratch tree]  Test accuracy: {acc:.4f}")

    # --- sklearn's tree, for sanity-check comparison ---
    sk_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
    sk_tree.fit(X_train, y_train)
    sk_acc = sk_tree.score(X_test, y_test)
    print(f"[sklearn tree]       Test accuracy: {sk_acc:.4f}")

    # --- Show which feature the root split on (interpretability!) ---
    root_feat = feature_names[tree.root.feature]
    print(f"\nRoot split feature : '{root_feat}'")
    print(f"Root split threshold: {tree.root.threshold:.3f}")

    # --- Overfitting demo: accuracy vs depth ---
    print("\nEffect of max_depth on train/test accuracy (over/underfitting):")
    print(f"{'depth':>6} | {'train_acc':>10} | {'test_acc':>10}")
    for depth in [1, 2, 3, 4, 6, 10, None]:
        t = DecisionTreeScratch(max_depth=depth if depth else 20)
        t.fit(X_train, y_train)
        train_acc = (t.predict(X_train) == y_train).mean()
        test_acc = (t.predict(X_test) == y_test).mean()
        print(f"{str(depth):>6} | {train_acc:>10.4f} | {test_acc:>10.4f}")


if __name__ == "__main__":
    main()
