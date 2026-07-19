"""
Chapter 11 (Day 11): Unsupervised Learning
A Course in Machine Learning (Hal Daume III), Chapter 13

From-scratch implementations of:
  1. K-Means (revisited) with the K-Means Convergence objective L(z, mu)
  2. Furthest-first heuristic and K-Means++ initialization (Section 13.1)
  3. Choosing K via the Bayes/Akaike Information Criteria (BIC / AIC)
  4. Principal Component Analysis (PCA) via the covariance-matrix
     eigendecomposition (Section 13.2), both the "maximum variance"
     and "reconstruction error" views

Tested on real datasets from sklearn.datasets and compared against
scikit-learn's reference implementations.
"""

import numpy as np
from sklearn.datasets import load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

RNG = np.random.RandomState(0)


# ==========================================================================
# 1. K-Means objective L(z, mu) (Eq. 13.1 in the book)
# ==========================================================================
def kmeans_objective(X, centers, labels):
    return np.sum((X - centers[labels]) ** 2)


# ==========================================================================
# 2a. Furthest-first heuristic (Section 13.1)
# ==========================================================================
def furthest_first_init(X, K, rng):
    N = X.shape[0]
    centers = np.empty((K, X.shape[1]))
    first = rng.randint(N)
    centers[0] = X[first]
    min_dist = np.linalg.norm(X - centers[0], axis=1) ** 2
    for k in range(1, K):
        next_idx = np.argmax(min_dist)         # farthest point from all chosen centers
        centers[k] = X[next_idx]
        dist_k = np.linalg.norm(X - centers[k], axis=1) ** 2
        min_dist = np.minimum(min_dist, dist_k)
    return centers


# ==========================================================================
# 2b. K-Means++ initialization (Algorithm 13.2, probabilistic version)
# ==========================================================================
def kmeans_pp_init(X, K, rng):
    N = X.shape[0]
    centers = np.empty((K, X.shape[1]))
    first = rng.randint(N)
    centers[0] = X[first]
    min_dist = np.linalg.norm(X - centers[0], axis=1) ** 2
    for k in range(1, K):
        probs = min_dist / min_dist.sum()
        next_idx = rng.choice(N, p=probs)      # sample proportional to distance^2
        centers[k] = X[next_idx]
        dist_k = np.linalg.norm(X - centers[k], axis=1) ** 2
        min_dist = np.minimum(min_dist, dist_k)
    return centers


# ==========================================================================
# 2c. K-Means main loop (Algorithm 13.1 / same as Chapter 2, Algorithm 4)
# ==========================================================================
def kmeans_fit(X, K, init="kmeans++", max_iter=300, seed=0):
    rng = np.random.RandomState(seed)
    if init == "random":
        idx = rng.choice(X.shape[0], K, replace=False)
        centers = X[idx].copy()
    elif init == "furthest-first":
        centers = furthest_first_init(X, K, rng)
    elif init == "kmeans++":
        centers = kmeans_pp_init(X, K, rng)
    else:
        raise ValueError(init)

    labels = np.zeros(X.shape[0], dtype=int)
    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = np.array([
            X[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centers[k]
            for k in range(K)
        ])
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers

    L = kmeans_objective(X, centers, labels)
    return centers, labels, L


# ==========================================================================
# 3. Choosing K via BIC / AIC (Section 13.1, Eq. 13.2 - 13.3)
# ==========================================================================
def bic_aic(X, K, L):
    N, D = X.shape
    bic = L + K * np.log(D)
    aic = L + 2 * K * D
    return bic, aic


# ==========================================================================
# 4. PCA from scratch (Section 13.2)
# ==========================================================================
class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # data covariance matrix, D x D  (Eq. 13.9's X^T X, up to a 1/N factor)
        cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
        eigvals, eigvecs = np.linalg.eigh(cov)     # eigh: cov is symmetric
        order = np.argsort(eigvals)[::-1]          # descending eigenvalue order
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        self.explained_variance_ = eigvals[: self.n_components]
        self.components_ = eigvecs[:, : self.n_components].T  # (K, D)
        total_var = eigvals.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def reconstruct(self, Z):
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        Z = self.transform(X)
        Xhat = self.reconstruct(Z)
        return np.mean(np.sum((X - Xhat) ** 2, axis=1))


# ==========================================================================
# MAIN EXPERIMENTS
# ==========================================================================
if __name__ == "__main__":

    # ----------------------------------------------------------------
    # Data for clustering: Wine dataset (3 known cultivars => 3 "true" clusters)
    # ----------------------------------------------------------------
    wine = load_wine()
    Xw, yw = wine.data, wine.target
    Xw_s = StandardScaler().fit_transform(Xw)

    print("=" * 72)
    print("EXPERIMENT A: initialization matters -- random vs furthest-first vs K-Means++")
    print("=" * 72)
    print(f"Dataset: Wine, {Xw.shape[0]} examples, {Xw.shape[1]} features, 3 true classes")
    print(f"{'init method':>16} | {'mean L (10 runs)':>17} | {'mean ARI':>9}")
    for init in ["random", "furthest-first", "kmeans++"]:
        Ls, aris = [], []
        for seed in range(10):
            centers, labels, L = kmeans_fit(Xw_s, K=3, init=init, seed=seed)
            Ls.append(L)
            aris.append(adjusted_rand_score(yw, labels))
        print(f"{init:>16} | {np.mean(Ls):>17.2f} | {np.mean(aris):>9.4f}")
    print("(Lower L = tighter clusters; higher ARI = closer match to the true")
    print(" cultivar labels. Smarter initialization gets consistently better")
    print(" objective values and truer clusters, on average, than pure random.)")

    print("\n" + "=" * 72)
    print("EXPERIMENT B: from-scratch K-Means++ vs sklearn's KMeans (init='k-means++')")
    print("=" * 72)
    my_centers, my_labels, my_L = kmeans_fit(Xw_s, K=3, init="kmeans++", seed=1)
    my_ari = adjusted_rand_score(yw, my_labels)

    sk_km = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=1).fit(Xw_s)
    sk_ari = adjusted_rand_score(yw, sk_km.labels_)

    print(f"From-scratch K-Means++  : L={my_L:.2f}, ARI={my_ari:.4f}")
    print(f"sklearn KMeans          : L={sk_km.inertia_:.2f}, ARI={sk_ari:.4f}")

    print("\n" + "=" * 72)
    print("EXPERIMENT C: choosing K via BIC / AIC (Section 13.1)")
    print("=" * 72)
    print(f"{'K':>4} | {'L (inertia)':>12} | {'BIC':>10} | {'AIC':>12}")
    for K in range(1, 8):
        best_L = min(kmeans_fit(Xw_s, K=K, init="kmeans++", seed=s)[2] for s in range(5))
        bic, aic = bic_aic(Xw_s, K, best_L)
        marker = "  <-- true K=3" if K == 3 else ""
        print(f"{K:>4} | {best_L:>12.2f} | {bic:>10.2f} | {aic:>12.2f}{marker}")
    print("(BIC/AIC penalize extra clusters; look for where L stops dropping")
    print(" 'enough' to justify the added K*D penalty term.)")

    # ----------------------------------------------------------------
    # Data for PCA: Digits dataset (8x8 images = 64 features)
    # ----------------------------------------------------------------
    print("\n" + "=" * 72)
    print("EXPERIMENT D: PCA from scratch vs sklearn.decomposition.PCA")
    print("=" * 72)
    digits = load_digits()
    Xd, yd = digits.data, digits.target
    print(f"Dataset: Digits, {Xd.shape[0]} examples, {Xd.shape[1]} features (8x8 images)")

    my_pca = PCAFromScratch(n_components=10).fit(Xd)
    sk_pca = PCA(n_components=10, svd_solver="full").fit(Xd)

    print(f"{'component':>10} | {'my var ratio':>13} | {'sklearn var ratio':>18}")
    for i in range(10):
        print(f"{i+1:>10} | {my_pca.explained_variance_ratio_[i]:>13.5f} | "
              f"{sk_pca.explained_variance_ratio_[i]:>18.5f}")

    my_cum = np.cumsum(my_pca.explained_variance_ratio_)[-1]
    sk_cum = np.cumsum(sk_pca.explained_variance_ratio_)[-1]
    print(f"\nCumulative variance explained by 10 components: "
          f"mine={my_cum:.4f}, sklearn={sk_cum:.4f}")

    # Sign of eigenvectors is arbitrary; compare magnitude of correlation instead
    my_Z = my_pca.transform(Xd)
    sk_Z = sk_pca.transform(Xd)
    corrs = [abs(np.corrcoef(my_Z[:, i], sk_Z[:, i])[0, 1]) for i in range(10)]
    print(f"Per-component |correlation| between my projection and sklearn's: "
          f"min={min(corrs):.4f}, mean={np.mean(corrs):.4f}")
    print("(Correlation ~1.0 confirms the two implementations find the same")
    print(" principal axes, up to an arbitrary sign flip per eigenvector.)")

    print("\n" + "=" * 72)
    print("EXPERIMENT E: reconstruction error vs number of components (Eq. 13.14)")
    print("=" * 72)
    print(f"{'K components':>13} | {'reconstruction err':>19} | {'cum. var. explained':>19}")
    for K in [1, 2, 5, 10, 20, 40, 64]:
        pca_k = PCAFromScratch(n_components=K).fit(Xd)
        err = pca_k.reconstruction_error(Xd)
        cum_var = np.cumsum(pca_k.explained_variance_ratio_)[-1]
        print(f"{K:>13} | {err:>19.4f} | {cum_var:>19.4f}")
    print("(As K -> D=64, reconstruction error -> 0 and variance explained -> 1,")
    print(" confirming that maximizing variance and minimizing reconstruction")
    print(" error are the same objective, as derived in Section 13.2.)")
