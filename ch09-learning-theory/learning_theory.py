"""
Chapter 9: Learning Theory
A Course in Machine Learning (Hal Daume III)

From-scratch implementation and empirical demonstration of:
  1. The "Throw Out Bad Terms" algorithm for PAC-learning boolean
     conjunctions (Algorithm 10.4), with a Monte-Carlo estimate of its
     (epsilon, delta)-PAC behaviour as a function of sample size N.
  2. VC dimension of linear classifiers in 2D: showing 3 points CAN be
     shattered by a linear classifier, but a specific set of 4 points
     CANNOT (Section 10.6).
  3. Occam's Razor in practice: comparing the sample complexity /
     generalization behaviour of a small hypothesis class (decision
     stumps) against a large one (unrestricted decision trees) on a
     real dataset (Section 10.5).

Real feature distributions come from sklearn's Breast Cancer Wisconsin
dataset throughout; only the *ground-truth labeling rule* in Experiment A
is hand-specified, because PAC sample-complexity theorems are statements
about learning a *known, fixed* target concept -- you need to know the
truth to measure "true" generalization error exactly, which no public
dataset's real labels allow us to do.
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

RNG = np.random.RandomState(0)


# ==========================================================================
# Experiment A: PAC-learning boolean conjunctions ("Throw Out Bad Terms")
# ==========================================================================
def binary_conjunction_train(X, y):
    """
    Algorithm 10.4 (BinaryConjunctionTrain). X is (N, D) binary, y in {0,1}.
    Returns a set of "surviving" literals: (feature, required_value) pairs
    that are consistent with every positive example seen.
    """
    D = X.shape[1]
    # start with every possible literal: "x_d = 0" and "x_d = 1"
    surviving = {(d, v) for d in range(D) for v in (0, 1)}
    for xi, yi in zip(X, y):
        if yi == 1:  # only positive examples throw out bad terms
            for d in range(D):
                literal_kept = (d, xi[d])
                literal_dropped = (d, 1 - xi[d])
                surviving.discard(literal_dropped)
    return surviving


def binary_conjunction_predict(surviving, X):
    """A point is positive iff it satisfies every surviving literal."""
    preds = np.ones(X.shape[0], dtype=int)
    for (d, v) in surviving:
        preds &= (X[:, d] == v)
    return preds


print("=" * 70)
print("EXPERIMENT A: PAC-learning a boolean conjunction (Algorithm 10.4)")
print("=" * 70)

data = load_breast_cancer()
X_raw = data.data
medians = np.median(X_raw, axis=0)
X_bin_full = (X_raw > medians).astype(int)  # real feature distribution, binarized
D = X_bin_full.shape[1]
print(f"Using {X_bin_full.shape[0]} real (binarized) feature vectors, D={D} boolean features")

# Ground-truth concept: a conjunction over 3 of the real binarized features.
# (We must fix a KNOWN concept to measure true generalization error exactly --
# this is what the PAC framework's "e" and "delta" are defined against.)
true_literals = {(2, 1), (5, 0), (7, 1)}
y_true_full = binary_conjunction_predict(true_literals, X_bin_full)
print(f"Ground-truth concept: literals {true_literals}")
print(f"Positive rate under this concept: {y_true_full.mean():.3f}")

# Held-out "distribution" sample to estimate true error e (large, fixed)
X_pool, X_eval, y_pool, y_eval = train_test_split(
    X_bin_full, y_true_full, test_size=200, random_state=1
)

print(f"\n{'N (train size)':>15} | {'mean test err':>13} | {'P(err > 0.05)':>13}  (over 200 trials)")
print("-" * 60)
for N in [5, 10, 20, 40, 80, 160, 300]:
    errs = []
    for trial in range(200):
        rng = np.random.RandomState(trial)
        idx = rng.choice(len(X_pool), size=min(N, len(X_pool)), replace=False)
        Xs, ys = X_pool[idx], y_pool[idx]
        surviving = binary_conjunction_train(Xs, ys)
        preds = binary_conjunction_predict(surviving, X_eval)
        err = np.mean(preds != y_eval)
        errs.append(err)
    errs = np.array(errs)
    print(f"{N:>15} | {errs.mean():>13.4f} | {np.mean(errs > 0.05):>13.4f}")

print("\nAs N grows, both the mean test error and the fraction of 'bad' runs")
print("(error > 5%) shrink towards zero -- exactly the (epsilon, delta)-PAC")
print("guarantee of Theorem 13: with enough examples, the learned conjunction")
print("is *probably* (high 1-delta) *approximately* (low error e) correct.")
print(f"(Occam's bound for this H, |H|=4^D=4^{D}, gives a very loose worst-case")
print(f" sample complexity -- in practice this simple algorithm converges far")
print(f" faster than that pessimistic bound suggests.)")


# ==========================================================================
# Experiment B: VC dimension of linear classifiers in 2D (Section 10.6)
# ==========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT B: VC dimension of linear classifiers in the plane")
print("=" * 70)


def can_shatter(points, max_iter=200):
    """
    For every possible binary labeling of `points`, check whether SOME
    linear classifier (trained with the perceptron) can fit it perfectly.
    Returns True only if ALL 2^n labelings can be fit (i.e. the points
    are shattered).
    """
    n = len(points)
    for labels_bits in range(2 ** n):
        y = np.array([(labels_bits >> i) & 1 for i in range(n)])
        if len(np.unique(y)) < 2:
            continue  # a constant labeling is trivially linearly separable
        clf = Perceptron(max_iter=max_iter, tol=1e-3, random_state=0)
        clf.fit(points, y)
        preds = clf.predict(points)
        if not np.array_equal(preds, y):
            return False, y  # found a labeling that cannot be fit
    return True, None


# 3 points in "general position" (a triangle) -- book claims this can always
# be shattered by a linear classifier
triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
shattered_3, bad_labeling_3 = can_shatter(triangle)
print(f"Can a linear classifier shatter 3 points (triangle)?  {shattered_3}")

# 4 points arranged as XOR corners -- classic example of 4 points that
# CANNOT be shattered (the XOR-style labeling defeats every line)
square = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
shattered_4, bad_labeling_4 = can_shatter(square)
print(f"Can a linear classifier shatter 4 points (unit square)? {shattered_4}")
if not shattered_4:
    print(f"  Counter-example labeling that fails: {bad_labeling_4}  "
          f"(this is the XOR pattern: opposite corners share a label)")

print("\nThis matches Theorem in Section 10.6: the VC dimension of linear")
print("classifiers in 2D is exactly 3 -- some set of 3 points can always be")
print("shattered, but every set of 4 points has at least one labeling (XOR)")
print("that no line can separate.")


# ==========================================================================
# Experiment C: Occam's Razor in practice -- small vs large hypothesis
# class, sample complexity on REAL data (Section 10.5)
# ==========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT C: Occam's Razor -- decision stump vs unrestricted tree")
print("=" * 70)

y = np.where(data.target == 0, 0, 1)
X = data.data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Full training pool: {len(X_train_full)}  |  fixed test set: {len(X_test)}")

print(f"\n{'N':>5} | {'stump train':>11} | {'stump test':>10} | {'full-tree train':>16} | {'full-tree test':>15}")
print("-" * 72)
for N in [10, 20, 40, 80, 160, 300, len(X_train_full)]:
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_train_full), size=N, replace=False)
    Xs, ys = X_train_full[idx], y_train_full[idx]

    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(Xs, ys)
    full_tree = DecisionTreeClassifier(max_depth=None, random_state=0).fit(Xs, ys)

    stump_train = accuracy_score(ys, stump.predict(Xs))
    stump_test = accuracy_score(y_test, stump.predict(X_test))
    tree_train = accuracy_score(ys, full_tree.predict(Xs))
    tree_test = accuracy_score(y_test, full_tree.predict(X_test))
    print(f"{N:>5} | {stump_train:>11.4f} | {stump_test:>10.4f} | {tree_train:>16.4f} | {tree_test:>15.4f}")

print("\nThe unrestricted tree always reaches ~100% TRAINING accuracy, at every N")
print("(it can memorize any training set, however small or large) -- notice")
print("its train/test gap never closes. The stump's train accuracy actually")
print("DROPS as N grows (it runs out of room to fit more points with just one")
print("split), and its train and test numbers stay close together throughout.")
print("This is Occam's point in miniature: a huge hypothesis class can always")
print("drive training error to zero, but that zero is not informative about")
print("test performance -- the persistent gap for the full tree is the")
print("symptom of a hypothesis class large enough to memorize whatever it")
print("sees, regardless of how many examples you give it.")
