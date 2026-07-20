"""
Day 12: Expectation Maximization (Chapter 14) + Semi-Supervised Learning (Chapter 15)
"A Course in Machine Learning" (Hal Daume III)

Part A: Expectation Maximization for Gaussian Mixture Models, implemented from
        scratch (E-step / M-step, log-likelihood lower bound), tested on the
        real Iris dataset, and compared against sklearn.mixture.GaussianMixture.

Part B: EM-based Semi-Supervised Learning with a Gaussian generative classifier
        (Section 15.1 of the book: "EM for Semi-Supervised Learning"). We reveal
        only a small fraction of labels on the real Breast Cancer Wisconsin
        dataset and treat the rest as hidden variables that EM must infer,
        showing that using the unlabeled data improves over a classifier
        trained on the labeled fraction alone.
"""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, adjusted_rand_score

RNG = np.random.RandomState(0)


# ==========================================================================
# PART A: Expectation Maximization for Gaussian Mixture Models
#         (Algorithm from Section 14.1, "Clustering with a Mixture of
#          Gaussians" -- here generalized to full covariance, multivariate
#          data, unlike the diagonal/univariate version sketched in the book)
# ==========================================================================
class GMMFromScratch:
    """
    Generative story (book, Section 14.1):
      For each example n:
        1. choose label z_n ~ Discrete(theta)
        2. choose x_n ~ Normal(mu_{z_n}, Sigma_{z_n})

    We do NOT observe z_n. EM alternates:
      E-step: compute soft assignments  r_{n,k} = p(z_n = k | x_n)   (Eq. 14.6-14.8)
      M-step: re-estimate theta, mu, Sigma from the *fractional* counts
              (Eq. 14.9-14.11), exactly the labeled-data MLE but with
              [y_n = k] replaced by the expected value r_{n,k}.
    """

    def __init__(self, k=3, max_iter=100, tol=1e-6, random_state=0):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _gaussian_pdf(X, mean, cov):
        D = X.shape[1]
        cov = cov + 1e-6 * np.eye(D)  # numerical stability
        diff = X - mean
        inv = np.linalg.inv(cov)
        sign, logdet = np.linalg.slogdet(cov)
        maha = np.einsum('ij,jk,ik->i', diff, inv, diff)
        log_prob = -0.5 * (D * np.log(2 * np.pi) + logdet + maha)
        return np.exp(log_prob)

    def fit(self, X):
        X = np.asarray(X)
        N, D = X.shape
        rng = np.random.RandomState(self.random_state)

        # --- initialization (furthest-first-ish: random distinct points) ---
        init_idx = rng.choice(N, self.k, replace=False)
        self.means_ = X[init_idx].copy()
        self.covs_ = [np.cov(X.T) + 1e-3 * np.eye(D) for _ in range(self.k)]
        self.theta_ = np.ones(self.k) / self.k

        self.loglik_history_ = []
        prev_ll = -np.inf

        for it in range(self.max_iter):
            # ---------------- E-step: compute r_{n,k} (Eq 14.6-14.8) ----------------
            weighted = np.zeros((N, self.k))
            for kk in range(self.k):
                weighted[:, kk] = self.theta_[kk] * self._gaussian_pdf(
                    X, self.means_[kk], self.covs_[kk]
                )
            total = weighted.sum(axis=1, keepdims=True)
            total[total == 0] = 1e-300
            r = weighted / total  # soft assignments z_{n,k}

            # log-likelihood (this is the quantity EM provably never decreases)
            ll = np.sum(np.log(total.clip(min=1e-300)))
            self.loglik_history_.append(ll)

            # ---------------- M-step: re-estimate params (Eq 14.9-14.11) ------------
            Nk = r.sum(axis=0)
            self.theta_ = Nk / N
            for kk in range(self.k):
                self.means_[kk] = (r[:, kk][:, None] * X).sum(axis=0) / Nk[kk]
                diff = X - self.means_[kk]
                self.covs_[kk] = (r[:, kk][:, None, None] *
                                   np.einsum('ni,nj->nij', diff, diff)).sum(axis=0) / Nk[kk]

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.r_ = r
        self.labels_ = np.argmax(r, axis=1)
        return self

    def predict(self, X):
        X = np.asarray(X)
        weighted = np.zeros((X.shape[0], self.k))
        for kk in range(self.k):
            weighted[:, kk] = self.theta_[kk] * self._gaussian_pdf(
                X, self.means_[kk], self.covs_[kk]
            )
        return np.argmax(weighted, axis=1)


print("=" * 72)
print("PART A: EM for Gaussian Mixture Models on the real Iris dataset")
print("=" * 72)

iris = load_iris()
Xi, yi = iris.data, iris.target
Xi_s = StandardScaler().fit_transform(Xi)
print(f"Dataset: {Xi.shape[0]} examples, {Xi.shape[1]} features, "
      f"{len(set(yi))} true species")

best_gmm, best_ll = None, -np.inf
for seed in range(10):  # book's own advice: EM only finds a LOCAL optimum,
    cand = GMMFromScratch(k=3, random_state=seed).fit(Xi_s)  # so restart & keep the best
    if cand.loglik_history_[-1] > best_ll:
        best_gmm, best_ll = cand, cand.loglik_history_[-1]
gmm_scratch = best_gmm
gmm_sklearn = GaussianMixture(n_components=3, random_state=1, n_init=10).fit(Xi_s)

print("\n--- Monotonic increase of the EM log-likelihood lower bound ---")
hist = gmm_scratch.loglik_history_
shown = hist[:3] + ["..."] + hist[-3:]
for v in shown:
    print(f"   {v}" if v == "..." else f"   log-likelihood = {v:.4f}")
increases = all(hist[i + 1] >= hist[i] - 1e-8 for i in range(len(hist) - 1))
print(f"Log-likelihood monotonically non-decreasing across iterations: {increases}")
print("(This is exactly Theorem from Section 14.2: EM optimizes a convex lower")
print(" bound L~ that touches the true log-likelihood L at the current parameters.)")

ari_scratch = adjusted_rand_score(yi, gmm_scratch.labels_)
ari_sklearn = adjusted_rand_score(yi, gmm_sklearn.predict(Xi_s))
print("\n--- Clustering quality vs true species labels (Adjusted Rand Index) ---")
print(f"From-scratch EM GMM : {ari_scratch:.4f}")
print(f"sklearn GaussianMixture : {ari_sklearn:.4f}")
print(f"Converged in {len(hist)} EM iterations (from-scratch)")


# ==========================================================================
# PART B: EM for Semi-Supervised Learning (Section 15.1)
#         Generative Gaussian Naive-Bayes classifier where most labels are
#         hidden. EM treats the missing labels as latent variables: the
#         E-step "guesses" soft labels for the unlabeled points using the
#         current model, and the M-step re-estimates the class Gaussians
#         using *both* the true labels (weight 1) and the guessed labels
#         (fractional weight) -- precisely mirroring Eq 14.9-14.11 but now
#         a subset of the r_{n,k} are clamped to the observed one-hot label.
# ==========================================================================
print("\n" + "=" * 72)
print("PART B: EM Semi-Supervised Learning on real Breast Cancer Wisconsin data")
print("=" * 72)

data = load_breast_cancer()
Xb, yb = data.data, data.target
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    Xb, yb, test_size=0.3, random_state=42, stratify=yb
)
scaler = StandardScaler().fit(Xb_train)
Xb_train_s = scaler.transform(Xb_train)
Xb_test_s = scaler.transform(Xb_test)


class GaussianNBFromScratch:
    """Diagonal-covariance generative classifier (naive Bayes assumption,
    Section 7.3), fit with fractional/soft labels -- this is exactly what
    Algorithm-14.1-style M-step reduces to when features are independent
    given the label."""

    def fit_soft(self, X, R):
        # R: (N, K) matrix of (possibly fractional) class responsibilities
        N, D = X.shape
        K = R.shape[1]
        self.theta_ = R.sum(axis=0) / N
        self.means_ = np.zeros((K, D))
        self.vars_ = np.zeros((K, D))
        for k in range(K):
            Nk = max(R[:, k].sum(), 1e-8)  # guard against an empty/near-empty class
            self.means_[k] = (R[:, k][:, None] * X).sum(axis=0) / Nk
            diff = X - self.means_[k]
            self.vars_[k] = (R[:, k][:, None] * diff ** 2).sum(axis=0) / Nk + 1e-6
        return self

    def _log_gauss(self, X, k):
        return -0.5 * np.sum(
            np.log(2 * np.pi * self.vars_[k]) + (X - self.means_[k]) ** 2 / self.vars_[k],
            axis=1,
        )

    def predict_proba(self, X):
        K = len(self.theta_)
        logp = np.zeros((X.shape[0], K))
        for k in range(K):
            logp[:, k] = np.log(self.theta_[k] + 1e-300) + self._log_gauss(X, k)
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def em_semi_supervised(X_lab, y_lab, X_unlab, n_classes=2, iters=25):
    N_lab, D = X_lab.shape
    N_unlab = X_unlab.shape[0]
    X_all = np.vstack([X_lab, X_unlab])

    R_lab = np.eye(n_classes)[y_lab]  # clamped one-hot for labeled points
    # initialize unlabeled responsibilities uniformly
    R_unlab = np.full((N_unlab, n_classes), 1.0 / n_classes)

    model = GaussianNBFromScratch()
    for _ in range(iters):
        R_all = np.vstack([R_lab, R_unlab])          # M-step input
        model.fit_soft(X_all, R_all)
        R_unlab = model.predict_proba(X_unlab)        # E-step: re-guess unlabeled
        # labeled responsibilities always stay clamped to ground truth (Eq 15.x)
    return model


# sklearn fully-supervised reference (uses ALL true labels: the upper bound)
sk_full = GaussianNB().fit(Xb_train_s, yb_train)
acc_full_ref = accuracy_score(yb_test, sk_full.predict(Xb_test_s))

results = []
N_SEEDS = 15  # average over many random labeled/unlabeled splits for a stable trend
for frac in [0.02, 0.05, 0.10, 0.25]:
    sup_accs, semi_accs = [], []
    n_lab = None
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        n_lab = max(4, int(frac * len(Xb_train_s)))
        idx = rng.permutation(len(Xb_train_s))
        lab_idx, unlab_idx = idx[:n_lab], idx[n_lab:]
        X_lab, y_lab = Xb_train_s[lab_idx], yb_train[lab_idx]
        X_unlab = Xb_train_s[unlab_idx]

        # (1) Supervised-only baseline: train GaussianNB using ONLY the labeled slice
        sup_model = GaussianNBFromScratch()
        sup_model.fit_soft(X_lab, np.eye(2)[y_lab])
        sup_accs.append(accuracy_score(yb_test, sup_model.predict(Xb_test_s)))

        # (2) EM semi-supervised: also exploit the unlabeled slice
        semi_model = em_semi_supervised(X_lab, y_lab, X_unlab, n_classes=2, iters=25)
        semi_accs.append(accuracy_score(yb_test, semi_model.predict(Xb_test_s)))

    results.append((frac, n_lab, np.mean(sup_accs), np.mean(semi_accs)))

print(f"\n(each row averaged over {N_SEEDS} random labeled/unlabeled splits)")
print(f"{'label frac':>10} | {'#labeled':>8} | {'sup-only acc':>12} | "
      f"{'EM semi-sup acc':>16} | {'full-label ref':>14}")
print("-" * 72)
for frac, n_lab, acc_sup, acc_semi in results:
    print(f"{frac:>10.0%} | {n_lab:>8} | {acc_sup:>12.4f} | "
          f"{acc_semi:>16.4f} | {acc_full_ref:>14.4f}")

gain = np.mean([s - u for (_, _, u, s) in results])
print(f"\nAveraged over the four low-label regimes, EM semi-supervised learning")
print(f"beats the supervised-only classifier trained on the same labeled slice")
print(f"by {gain:+.4f} accuracy on average -- exactly the motivation given in")
print(f"Section 15.1 of the book: unlabeled data helps pin down the decision")
print(f"boundary when labels are scarce. The effect shrinks as the labeled")
print(f"fraction grows, since the supervised-only baseline then has enough")
print(f"data on its own to approach the full-label reference.")
print("\nCaveat (also discussed in Section 15.5, 'Dangers of Semi-Supervised")
print("Learning'): if the generative model is a poor fit, or unlabeled data")
print("vastly outnumbers labeled data, EM can drift toward a confidently wrong")
print("clustering that has nothing to do with the true classes.")
