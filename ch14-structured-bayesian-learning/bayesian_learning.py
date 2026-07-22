"""
Chapter 19: Bayesian Learning
A Course in Machine Learning (Hal Daume III)

The book's own Chapter 19 is only an outline. This script builds out
the core Bayesian learning ideas that were already introduced in
Chapter 7 (Section 7.7, "Regularization via Priors") and pushes them
to their natural conclusion: instead of a single point estimate for
the weights, keep the FULL posterior distribution, so that every
prediction comes with a principled uncertainty estimate.

Two from-scratch demos, both on real data:

  1. Beta-Binomial conjugate update: estimate the malignancy rate of
     tumors in the real Breast Cancer Wisconsin dataset, showing how
     the posterior sharpens as more data arrives (this is exactly the
     coin-flipping example from Section 7.2/7.7, but on real labels).

  2. Bayesian Linear Regression (closed-form Gaussian posterior) on
     the real Diabetes dataset, compared against:
       (a) plain ridge regression (Ch. 6.6, the MAP/point-estimate view)
       (b) scikit-learn's BayesianRidge (reference implementation)
     showing that the Bayesian posterior mean recovers the same point
     predictions as ridge, while ALSO providing predictive variances.
"""

import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# ==========================================================================
# PART 1: Beta-Binomial conjugate estimation (Sections 7.2 / 7.7)
#
# Model: y_n ~ Bernoulli(beta), beta ~ Beta(a0, b0)  (prior belief)
# Posterior after seeing H "heads" (malignant) and T "tails" (benign):
#     beta | data ~ Beta(a0 + H, b0 + T)
# This is the exact Bayesian analogue of the maximum-likelihood coin
# estimate beta_MLE = H / (H + T) derived in Section 7.2, but it also
# gives a full distribution over beta, not just a point estimate.
# ==========================================================================
print("=" * 70)
print("PART 1: Beta-Binomial posterior for tumor malignancy rate")
print("        (real Breast Cancer Wisconsin labels)")
print("=" * 70)

cancer = load_breast_cancer()
# In sklearn's encoding, 0 = malignant, 1 = benign. Define "malignant" as
# our Bernoulli event of interest.
malignant = (cancer.target == 0).astype(int)
N_total = len(malignant)
print(f"Total examples: {N_total}  (true malignant rate = {malignant.mean():.4f})")

a0, b0 = 1.0, 1.0  # uniform (uninformative) Beta(1,1) prior
rng = np.random.RandomState(0)
order = rng.permutation(N_total)
stream = malignant[order]

checkpoints = [5, 20, 100, 300, N_total]
print(f"\n{'n seen':>8} | {'MLE (H/n)':>10} | {'posterior mean':>15} | {'95% credible interval':>24}")
print("-" * 70)
H_running = 0
seen = 0
for cp in checkpoints:
    H_running = stream[:cp].sum()
    seen = cp
    T_running = seen - H_running
    a_post, b_post = a0 + H_running, b0 + T_running
    post_mean = a_post / (a_post + b_post)
    mle = H_running / seen
    lo, hi = stats.beta.ppf([0.025, 0.975], a_post, b_post)
    print(f"{seen:>8} | {mle:>10.4f} | {post_mean:>15.4f} | [{lo:.4f}, {hi:.4f}]")

print("\nNote how the credible interval shrinks as more data arrives, and")
print("the posterior mean converges to the true rate, while remaining a")
print("full DISTRIBUTION over the malignancy rate rather than one number.")

# ==========================================================================
# PART 2: Bayesian Linear Regression, from scratch, closed form.
#
# Model (Section 7.6, extended to keep the posterior instead of just its
# mode):
#     y = w . x + eps,   eps ~ Nor(0, sigma^2)
#     w  ~ Nor(0, tau^2 I)                       (Gaussian prior, Sec 7.7)
#
# Posterior:  w | X, y ~ Nor(mu_N, Sigma_N)  with
#     Sigma_N = ( (1/sigma^2) X^T X + (1/tau^2) I )^-1
#     mu_N    = (1/sigma^2) Sigma_N X^T y
#
# Note mu_N is EXACTLY the ridge-regression solution from Section 6.6
# with lambda = sigma^2 / tau^2 -- Bayesian MAP estimation and
# regularized least squares are the same computation viewed two ways.
# ==========================================================================
print("\n" + "=" * 70)
print("PART 2: Bayesian Linear Regression on the real Diabetes dataset")
print("=" * 70)

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(f"Dataset shape: {X.shape[0]} patients, {X.shape[1]} clinical features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
y_mean = y_train.mean()
y_train_c = y_train - y_mean  # center target so we can skip a bias term


class BayesianLinearRegression:
    """Closed-form Bayesian linear regression with a fixed Gaussian prior."""

    def __init__(self, sigma2=1.0, tau2=1.0):
        self.sigma2 = sigma2  # observation noise variance
        self.tau2 = tau2      # prior variance on each weight

    def fit(self, X, y):
        D = X.shape[1]
        A = (1.0 / self.sigma2) * (X.T @ X) + (1.0 / self.tau2) * np.eye(D)
        self.Sigma_N = np.linalg.inv(A)          # posterior covariance
        self.mu_N = (1.0 / self.sigma2) * self.Sigma_N @ X.T @ y  # posterior mean
        return self

    def predict(self, X, return_std=False):
        mean = X @ self.mu_N
        if not return_std:
            return mean
        # predictive variance = data noise + uncertainty in w
        var = self.sigma2 + np.einsum("nd,de,ne->n", X, self.Sigma_N, X)
        return mean, np.sqrt(var)


# Estimate sigma^2 empirically from a quick ridge fit's residuals, and
# fix tau^2 by cross-validated-ish heuristic (kept simple/from-scratch).
ridge_probe = Ridge(alpha=1.0).fit(X_train_s, y_train_c)
resid = y_train_c - ridge_probe.predict(X_train_s)
sigma2_hat = np.var(resid)
tau2_hat = 25.0  # weakly informative prior variance on standardized features

blr = BayesianLinearRegression(sigma2=sigma2_hat, tau2=tau2_hat).fit(X_train_s, y_train_c)
mean_pred, std_pred = blr.predict(X_test_s, return_std=True)
mean_pred_full = mean_pred + y_mean  # undo centering

# Equivalent point-estimate baseline: ridge regression, lambda = sigma2/tau2
lam = sigma2_hat / tau2_hat
ridge = Ridge(alpha=lam).fit(X_train_s, y_train_c)
ridge_pred = ridge.predict(X_test_s) + y_mean

# Reference implementation: scikit-learn's BayesianRidge (learns sigma^2,
# tau^2 automatically via evidence maximization, rather than fixing them)
bridge = BayesianRidge().fit(X_train_s, y_train_c)
bridge_pred = bridge.predict(X_test_s) + y_mean

rmse_blr = np.sqrt(mean_squared_error(y_test, mean_pred_full))
rmse_ridge = np.sqrt(mean_squared_error(y_test, ridge_pred))
rmse_bridge = np.sqrt(mean_squared_error(y_test, bridge_pred))

print(f"\n{'Model':<38} | {'test RMSE':>10}")
print("-" * 55)
print(f"{'Ridge regression (point estimate)':<38} | {rmse_ridge:>10.3f}")
print(f"{'Bayesian linear regression (posterior mean)':<38} | {rmse_blr:>10.3f}")
print(f"{'sklearn BayesianRidge (reference)':<38} | {rmse_bridge:>10.3f}")

agree = np.corrcoef(mean_pred_full, ridge_pred)[0, 1]
print(f"\nCorrelation between our Bayesian posterior mean and ridge point")
print(f"estimate: {agree:.6f}  (confirms mu_N == ridge solution, as derived above)")

print("\nFirst 5 test patients: predicted disease progression +/- 1 std dev")
print(f"{'true':>7} | {'pred mean':>10} | {'pred std':>9} | {'in mean+-2std?':>15}")
for i in range(5):
    lo = mean_pred_full[i] - 2 * std_pred[i]
    hi = mean_pred_full[i] + 2 * std_pred[i]
    inside = lo <= y_test[i] <= hi
    print(f"{y_test[i]:>7.1f} | {mean_pred_full[i]:>10.1f} | {std_pred[i]:>9.1f} | {str(inside):>15}")

coverage = np.mean(
    (y_test >= mean_pred_full - 2 * std_pred) & (y_test <= mean_pred_full + 2 * std_pred)
)
print(f"\nEmpirical coverage of the (mean +/- 2*std) interval on ALL test points: "
      f"{coverage:.2%}")
print("(For a well-calibrated Gaussian posterior this should be close to ~95%.)")
