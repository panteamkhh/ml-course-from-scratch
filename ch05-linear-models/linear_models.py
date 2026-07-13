"""
Chapter 6: Linear Models
A Course in Machine Learning (Hal Daume III)

From-scratch implementation of:
  1. Regularized linear classifier trained with (sub)gradient descent,
     supporting hinge loss (soft-margin SVM) and logistic loss (logistic
     regression), both with an L2 regularizer.
  2. Closed-form ridge regression (2-norm regularized squared loss).

Tested on real datasets from sklearn.datasets, compared against
scikit-learn's reference implementations (LogisticRegression, LinearSVC,
Ridge), and used to demonstrate the effect of the regularization
hyperparameter lambda on the train/test accuracy gap.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, mean_squared_error

RNG = np.random.RandomState(42)


# ----------------------------------------------------------------------
# 1. Generic regularized linear classifier trained with gradient descent
#    (Section 6.1 - 6.5: objective = loss + (lambda/2)||w||^2)
# ----------------------------------------------------------------------
def hinge_loss_grad(y, a):
    """Subgradient of hinge loss max(0, 1 - y*a) wrt activation a."""
    margin = y * a
    # dL/da = -y where margin < 1, else 0 (Eq 6.25)
    g = np.where(margin < 1, -y, 0.0)
    return g


def logistic_loss_grad(y, a):
    """Gradient of logistic loss log(1+exp(-y*a)) wrt activation a."""
    # dL/da = -y * sigmoid(-y*a); clip to avoid overflow in exp for
    # very confident (large-margin) predictions.
    z = np.clip(y * a, -30, 30)
    return -y / (1.0 + np.exp(z))


LOSSES = {
    "hinge": hinge_loss_grad,
    "logistic": logistic_loss_grad,
}


class LinearClassifierFromScratch:
    """
    min_w,b  (1/N) sum_n loss(y_n, w.x_n + b) + (lambda/2)||w||^2

    Trained with full-batch gradient descent (Algorithm from Sec 6.4),
    with a shrinking step size eta_k = eta0 / sqrt(k).
    """

    def __init__(self, loss="hinge", lam=1e-2, eta0=1.0, max_iter=500):
        self.loss = loss
        self.lam = lam
        self.eta0 = eta0
        self.max_iter = max_iter

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        N, D = X.shape
        self.w = np.zeros(D)
        self.b = 0.0
        grad_fn = LOSSES[self.loss]

        for k in range(1, self.max_iter + 1):
            a = X @ self.w + self.b               # activations
            g_per_example = grad_fn(y, a)          # dL/da per example
            grad_w = (X.T @ g_per_example) / N + self.lam * self.w
            grad_b = g_per_example.mean()

            eta = self.eta0 / np.sqrt(k)
            self.w -= eta * grad_w
            self.b -= eta * grad_b

        return self

    def decision_function(self, X):
        return np.asarray(X) @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)


# ----------------------------------------------------------------------
# 2. Closed-form ridge regression (Section 6.6)
#    w = (X^T X + lambda I)^-1 X^T y
# ----------------------------------------------------------------------
class RidgeRegressionFromScratch:
    """
    Closed-form ridge regression with an (unregularized) intercept.
    We center X and y so that the bias term never needs to be
    regularized (matching sklearn's default fit_intercept=True), then
    solve  w = (X^T X + lambda I)^-1 X^T y  on the centered data.
    """

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        self.x_mean = X.mean(axis=0)
        self.y_mean = y.mean()
        Xc = X - self.x_mean
        yc = y - self.y_mean
        D = X.shape[1]
        A = Xc.T @ Xc + self.lam * np.eye(D)
        self.w = np.linalg.solve(A, Xc.T @ yc)
        self.b = self.y_mean - self.x_mean @ self.w
        return self

    def predict(self, X):
        return np.asarray(X) @ self.w + self.b


# ----------------------------------------------------------------------
# Experiment A: Hinge loss (SVM) vs sklearn LinearSVC
# ----------------------------------------------------------------------
print("=" * 70)
print("EXPERIMENT A: From-scratch linear SVM (hinge loss) vs sklearn LinearSVC")
print("=" * 70)

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)   # +-1 labels, as the book uses

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

svm_scratch = LinearClassifierFromScratch(loss="hinge", lam=1e-2, eta0=1.0, max_iter=1000)
svm_scratch.fit(X_train_s, y_train)
pred_scratch = svm_scratch.predict(X_test_s)
acc_scratch = accuracy_score(y_test, pred_scratch)

sk_svm = LinearSVC(C=1.0 / (1e-2), max_iter=5000, random_state=42)
sk_svm.fit(X_train_s, y_train)
acc_sklearn = accuracy_score(y_test, sk_svm.predict(X_test_s))

agreement = np.mean(pred_scratch == sk_svm.predict(X_test_s))
print(f"From-scratch hinge-loss SVM test accuracy : {acc_scratch:.4f}")
print(f"sklearn LinearSVC        test accuracy     : {acc_sklearn:.4f}")
print(f"Prediction agreement rate                  : {agreement:.4f}")

# ----------------------------------------------------------------------
# Experiment B: Logistic loss vs sklearn LogisticRegression
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT B: From-scratch logistic regression vs sklearn LogisticRegression")
print("=" * 70)

logreg_scratch = LinearClassifierFromScratch(loss="logistic", lam=1e-2, eta0=1.0, max_iter=1000)
logreg_scratch.fit(X_train_s, y_train)
pred_log_scratch = logreg_scratch.predict(X_test_s)
acc_log_scratch = accuracy_score(y_test, pred_log_scratch)

sk_logreg = LogisticRegression(C=1.0 / (1e-2), max_iter=5000)
sk_logreg.fit(X_train_s, y_train)
acc_log_sklearn = accuracy_score(y_test, sk_logreg.predict(X_test_s))

agreement_log = np.mean(pred_log_scratch == sk_logreg.predict(X_test_s))
print(f"From-scratch logistic-loss classifier test accuracy : {acc_log_scratch:.4f}")
print(f"sklearn LogisticRegression       test accuracy       : {acc_log_sklearn:.4f}")
print(f"Prediction agreement rate                            : {agreement_log:.4f}")

# ----------------------------------------------------------------------
# Experiment C: Effect of regularization strength lambda (Section 6.3)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT C: Regularization strength (lambda) vs train/test accuracy")
print("=" * 70)
print(f"{'lambda':>10} | {'train acc':>10} | {'test acc':>9} | {'||w||':>8}")
print("-" * 45)
for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    clf = LinearClassifierFromScratch(loss="logistic", lam=lam, eta0=1.0, max_iter=1000)
    clf.fit(X_train_s, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train_s))
    test_acc = accuracy_score(y_test, clf.predict(X_test_s))
    print(f"{lam:>10.4f} | {train_acc:>10.4f} | {test_acc:>9.4f} | {np.linalg.norm(clf.w):>8.4f}")

print("\nAs lambda grows, ||w|| shrinks (Section 6.3: small weights <=> simple")
print("functions), training accuracy drops slightly, but test accuracy stays")
print("robust until lambda gets too large, at which point the model underfits.")

# ----------------------------------------------------------------------
# Experiment D: Closed-form Ridge Regression vs sklearn Ridge
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("EXPERIMENT D: Closed-form ridge regression vs sklearn Ridge")
print("(Diabetes dataset - real regression data)")
print("=" * 70)

housing = load_diabetes()
Xh, yh = housing.data, housing.target
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    Xh, yh, test_size=0.3, random_state=42
)
h_scaler = StandardScaler().fit(Xh_train)
Xh_train_s = h_scaler.transform(Xh_train)
Xh_test_s = h_scaler.transform(Xh_test)

lam_ridge = 0.5
ridge_scratch = RidgeRegressionFromScratch(lam=lam_ridge).fit(Xh_train_s, yh_train)
pred_ridge_scratch = ridge_scratch.predict(Xh_test_s)
mse_scratch = mean_squared_error(yh_test, pred_ridge_scratch)

sk_ridge = Ridge(alpha=lam_ridge).fit(Xh_train_s, yh_train)
mse_sklearn = mean_squared_error(yh_test, sk_ridge.predict(Xh_test_s))

print(f"Dataset shape: {Xh.shape[0]} examples, {Xh.shape[1]} features")
print(f"From-scratch closed-form ridge   test MSE : {mse_scratch:.4f}")
print(f"sklearn Ridge                    test MSE : {mse_sklearn:.4f}")
print(f"Max abs weight difference                : {np.max(np.abs(ridge_scratch.w - sk_ridge.coef_)):.6f}")

print("\nThe closed-form solution w = (X^T X + lambda*I)^-1 X^T y matches")
print("sklearn's Ridge almost exactly (up to floating point / solver differences),")
print("confirming the derivation in Section 6.6.")
