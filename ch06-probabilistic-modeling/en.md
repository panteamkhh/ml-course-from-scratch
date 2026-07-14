# Chapter 7: Probabilistic Modeling

> Instead of asking "what's the best hyperplane?", probabilistic modeling asks "what distribution most plausibly generated this data?" — and classification falls out for free.

**Type:** Learn + Build **Languages:** Python **Prerequisites:** Chapter 6 (Linear Models) **Time:** ~45 minutes
**Source:** A Course in Machine Learning, Hal Daumé III — Chapter 7

## Learning Objectives
- Define the generative story behind a Naive Bayes classifier, for both continuous (Gaussian) and binary (Bernoulli) features.
- Derive relative-frequency estimation as the solution to a constrained maximum-likelihood optimization problem.
- Implement Gaussian Naive Bayes and Bernoulli Naive Bayes from scratch and validate them against scikit-learn.
- Compare generative (Naive Bayes) and conditional/discriminative (logistic regression) models trained on the same linear decision boundary.
- Explain the classic Naive-Bayes-vs-logistic-regression data-size trade-off.

## The Problem
Linear models like the perceptron and SVM (Chapter 6) discriminate directly: they never model how the data was produced, only where the boundary between classes should sit. Probabilistic modeling takes a different philosophy: assume the data distribution `D(x, y)` exists, model it explicitly with a parametric family (e.g., Gaussian, Bernoulli), and estimate its parameters by maximum likelihood. If you knew `D` exactly, the *Bayes optimal classifier* — simply predicting `argmax_y D(x, y)` — would be provably optimal. Since you don't know `D`, you estimate it from data, and Naive Bayes is the simplest, most practical instance of this idea.

## The Concept

```mermaid
flowchart TD
    A[Assume: features are independent given the label] --> B[Estimate p(y) by counting label frequency]
    A --> C[Estimate p(x_d given y) per feature, per class]
    B --> D[Combine via Bayes rule at test time]
    C --> D
    D --> E[Predict argmax_y p(y) * product of p(x_d given y)]
```

- **The naive Bayes assumption**: `p(x_1, ..., x_D | y) = product_d p(x_d | y)`. This is almost always false in reality (words in a sentence are correlated!), but it turns an intractable joint distribution into a product of D easy one-dimensional estimation problems.
- **Different feature types, different per-feature models**: binary features → Bernoulli distribution (Section 7.3); continuous features → Gaussian distribution (Section 7.5); counts/categories → Multinomial/Discrete distribution.
- **Maximum likelihood estimation reduces to counting.** For Bernoulli features, `theta_{(y),d}` is just the fraction of class-`y` examples where feature `d` is present (Eq 7.21); no gradient descent required.
- **Naive Bayes' decision boundary is linear** (Section 7.4): the log-likelihood-ratio `log p(y=+1|x) - log p(y=-1|x)` reduces algebraically to `w·x + b`, exactly the same functional form as the perceptron, SVM, and logistic regression from Chapter 6 — only the *estimation procedure* differs (counting vs. gradient descent).

## Build It

**1. Gaussian Naive Bayes**: estimate a per-class, per-feature mean and variance, then use the Gaussian log-density as the log-likelihood:

```python
self.mu_[c] = Xc.mean(axis=0)
self.var_[c] = Xc.var(axis=0) + 1e-9
log_lik = -0.5*np.sum(np.log(2*np.pi*var)) - 0.5*np.sum((X-mu)**2/var, axis=1)
```

**2. Bernoulli Naive Bayes for text**: estimate `theta_d = P(word d present | class)` by Laplace-smoothed relative frequency (Eq 7.21), then score new documents with the log-likelihood in Eq (7.18):

```python
theta = (counts + alpha) / (len(Xc) + 2 * alpha)
scores[:, i] = log_prior[c] + X @ log_theta[c] + (1 - X) @ log_one_minus_theta[c]
```

**3. Compare against logistic regression** trained on the exact same bag-of-words features, to see the generative-vs-conditional trade-off in practice.

**Run it:**
```bash
python3 probabilistic_modeling.py
```
(Downloads the real, public [SMS Spam Collection dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv) into `sms.tsv` on first run — 5,574 real text messages labeled ham/spam.)

**Expected output (abridged, real run):**
```
EXPERIMENT A: Gaussian Naive Bayes on sklearn's Wine dataset
From-scratch Gaussian NB test accuracy : 1.0000
sklearn GaussianNB       test accuracy : 1.0000
Prediction agreement rate              : 1.0000

EXPERIMENT B: Bernoulli Naive Bayes for text (SMS Spam Collection)
Train messages: 4179, Test messages: 1393, Vocabulary size: 2000
From-scratch Bernoulli NB test accuracy : 0.9842
sklearn BernoulliNB      test accuracy   : 0.9842
Prediction agreement rate                : 1.0000

EXPERIMENT C: Generative (Naive Bayes) vs Conditional (Logistic Regression)
Bernoulli Naive Bayes (ours)  |        0.9842
Logistic Regression (sklearn) |        0.9770

EXPERIMENT D: Naive Bayes vs Logistic Regression as training size grows
# train docs | NB test acc | LogReg test acc
          83 |      0.8658 |          0.8729
        1044 |      0.9541 |          0.9627
        4179 |      0.9842 |          0.9770
```
Both from-scratch models match sklearn's `GaussianNB`/`BernoulliNB` predictions exactly (100% agreement). Experiment D shows the classic generative-vs-discriminative story: with only 83 training messages both models are weak, but as data grows Naive Bayes' counting-based estimates stabilize quickly. On this particular spam-detection task the independence assumption is a fairly good match for the data (individual "spammy" words really are strong, largely independent signals), so Naive Bayes stays competitive with — and here even edges out — logistic regression even at full data size, which will not hold on every dataset.

## Use It

| API / Function | When to use it |
|---|---|
| `GaussianNaiveBayesFromScratch()` | Continuous features, want an extremely fast, closed-form generative baseline. |
| `BernoulliNaiveBayesFromScratch(alpha)` | Binary/bag-of-words text features; `alpha` controls Laplace smoothing strength. |
| `sklearn.naive_bayes.GaussianNB` / `BernoulliNB` | Production use; handles sparse matrices and more numerical edge cases. |
| `sklearn.linear_model.LogisticRegression` | When you suspect the independence assumption is badly violated, or want a discriminatively-trained linear boundary instead. |

## Exercises
1. Add Laplace smoothing (`alpha`) as a tunable hyperparameter to `GaussianNaiveBayesFromScratch`'s variance estimate (i.e., add `alpha` to `var_` directly) and see how it affects accuracy on classes with very few training examples.
2. Implement Multinomial Naive Bayes (using word *counts* rather than presence/absence) and compare it with the Bernoulli version on the SMS dataset.
3. Extract the top 10 words with the highest `log_theta[spam] - log_theta[ham]` and check whether they match your intuition about what makes a text message "spammy."

## Key Terms

| Term | Common Assumption | Precise Meaning |
|---|---|---|
| Naive Bayes | "A weak baseline you outgrow immediately" | A generative classifier whose only approximation is feature independence given the label; often extremely competitive on high-dimensional, sparse data like text. |
| Generative Model | "Models that generate images or text" | Any model that specifies `p(x, y)` (or `p(x | y)` and `p(y)`) rather than directly modeling `p(y | x)`; "generative" refers to being able to sample fictitious data, not to any particular data type. |
| Maximum Likelihood Estimation | "A complicated statistical procedure" | Choosing parameters that maximize the probability of the observed training data; for Naive Bayes this reduces to simple counting. |
| Bayes Optimal Classifier | "The best classifier we could ever build" | The theoretical classifier `argmax_y D(x,y)` that is optimal *if* you knew the true data distribution `D` exactly — a benchmark, not a practical algorithm, since `D` is always unknown. |
