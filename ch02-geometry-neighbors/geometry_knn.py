"""
Chapter 2: Geometry and Nearest Neighbors
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. K-Nearest Neighbors (KNN) classifier
  2. K-Means clustering

Tested on real datasets from sklearn.datasets, compared against
scikit-learn's reference implementations, and used to demonstrate
the underfitting/overfitting trade-off controlled by K, plus the
curse of dimensionality effect on distances.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

RNG = np.random.RandomState(42)


# ----------------------------------------------------------------------
# 1. K-Nearest Neighbors, implemented from scratch (Algorithm 3 in book)
# ----------------------------------------------------------------------
class KNNFromScratch:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        # KNN has no real "training" phase: just store the data
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.empty(X.shape[0], dtype=self.y_train.dtype)
        for i, x in enumerate(X):
            # Euclidean distance from x to every training point
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            nn_idx = np.argsort(dists)[: self.k]
            nn_labels = self.y_train[nn_idx]
            # majority vote among the k nearest neighbors
            values, counts = np.unique(nn_labels, return_counts=True)
            preds[i] = values[np.argmax(counts)]
        return preds


# ----------------------------------------------------------------------
# 2. K-Means clustering, implemented from scratch (Algorithm 4 in book)
# ----------------------------------------------------------------------
class KMeansFromScratch:
    def __init__(self, k=3, max_iter=100, random_state=0):
        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.RandomState(self.random_state)
        # furthest-first-ish init: pick k random distinct points as means
        init_idx = rng.choice(X.shape[0], self.k, replace=False)
        self.centers_ = X[init_idx].copy()

        for _ in range(self.max_iter):
            # assignment step: each point goes to its closest center
            dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
            labels = np.argmin(dists, axis=1)

            # update step: recompute means
            new_centers = np.empty_like(self.centers_)
            for k in range(self.k):
                members = X[labels == k]
                if len(members) > 0:
                    new_centers[k] = members.mean(axis=0)
                else:
                    new_centers[k] = self.centers_[k]  # keep empty cluster in place

            if np.allclose(new_centers, self.centers_):
                self.centers_ = new_centers
                break
            self.centers_ = new_centers

        self.labels_ = labels
        return self

    def predict(self, X):
        X = np.asarray(X)
        dists = np.linalg.norm(X[:, None, :] - self.centers_[None, :, :], axis=2)
        return np.argmin(dists, axis=1)


# ----------------------------------------------------------------------
# Experiment A: KNN classification on the Breast Cancer dataset (real data)
# ----------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT A: KNN on sklearn's Breast Cancer Wisconsin dataset")
print("=" * 70)

data = load_breast_cancer()
X, y = data.data, data.target
print(f"Dataset shape: {X.shape[0]} examples, {X.shape[1]} features")
print(f"Classes: {data.target_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Feature scaling matters a lot for KNN (Chapter 2, feature scale section)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

print("\n--- Correctness check: from-scratch KNN vs sklearn KNN (k=5) ---")
my_knn = KNNFromScratch(k=5).fit(X_train_s, y_train)
my_pred = my_knn.predict(X_test_s)
my_acc = accuracy_score(y_test, my_pred)

sk_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_s, y_train)
sk_pred = sk_knn.predict(X_test_s)
sk_acc = accuracy_score(y_test, sk_pred)

agreement = np.mean(my_pred == sk_pred)
print(f"From-scratch KNN accuracy : {my_acc:.4f}")
print(f"sklearn KNN accuracy      : {sk_acc:.4f}")
print(f"Prediction agreement rate : {agreement:.4f}  (fraction of identical predictions)")

# ----------------------------------------------------------------------
# Experiment B: underfitting/overfitting as a function of K
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT B: train/test accuracy vs K (underfitting <-> overfitting)")
print("=" * 70)
print(f"{'K':>4} | {'train acc':>10} | {'test acc':>9}")
print("-" * 30)
for k in [1, 2, 3, 5, 10, 20, 40, 80, 150]:
    knn = KNNFromScratch(k=k).fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, knn.predict(X_train_s))
    test_acc = accuracy_score(y_test, knn.predict(X_test_s))
    print(f"{k:>4} | {train_acc:>10.4f} | {test_acc:>9.4f}")

# ----------------------------------------------------------------------
# Experiment C: K-Means clustering on the Wine dataset (real data)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT C: K-Means on sklearn's Wine dataset")
print("=" * 70)

wine = load_wine()
Xw, yw = wine.data, wine.target
Xw_s = StandardScaler().fit_transform(Xw)
print(f"Dataset shape: {Xw.shape[0]} examples, {Xw.shape[1]} features, {len(set(yw))} true classes")

my_km = KMeansFromScratch(k=3, random_state=1).fit(Xw_s)
sk_km = KMeans(n_clusters=3, n_init=10, random_state=1).fit(Xw_s)

my_ari = adjusted_rand_score(yw, my_km.labels_)
sk_ari = adjusted_rand_score(yw, sk_km.labels_)

print(f"From-scratch K-Means Adjusted Rand Index vs true labels : {my_ari:.4f}")
print(f"sklearn K-Means      Adjusted Rand Index vs true labels : {sk_ari:.4f}")
print("(Adjusted Rand Index measures cluster/label agreement; 1.0 = perfect, 0.0 = random)")

# ----------------------------------------------------------------------
# Experiment D: curse of dimensionality -- distances concentrate
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT D: curse of dimensionality (average pairwise distances)")
print("=" * 70)
print(f"{'Dimensions':>10} | {'mean dist':>10} | {'std dist':>9} | {'std/mean':>9}")
print("-" * 50)
for D in [2, 8, 32, 128, 512, 2048]:
    pts = RNG.uniform(0, 1, size=(200, D))
    # sample of pairwise distances (not all pairs, for speed)
    idx_a = RNG.randint(0, 200, size=3000)
    idx_b = RNG.randint(0, 200, size=3000)
    d = np.linalg.norm(pts[idx_a] - pts[idx_b], axis=1)
    print(f"{D:>10} | {d.mean():>10.4f} | {d.std():>9.4f} | {d.std()/d.mean():>9.4f}")

print("\nAs predicted in the book (Section 2.5): the mean distance grows like")
print("sqrt(D)/3, while the relative variance (std/mean) shrinks as D grows,")
print("meaning distances all start to look the same in high dimensions.")
