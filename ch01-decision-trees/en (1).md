# Chapter 1: Decision Trees

> A decision tree is nothing more than a sequence of yes/no questions, arranged so that each question splits your data as cleanly as possible.

**Type:** Learn + Build **Languages:** Python **Prerequisites:** None — this is lesson 1 **Time:** ~45 minutes
**Source:** *A Course in Machine Learning*, Hal Daumé III — Chapter 1

## Learning Objectives

- Explain what "learning" means in a supervised setting (data → hypothesis → prediction)
- Compute Entropy and Information Gain to decide which feature to split on
- Implement a decision tree classifier from scratch using recursive splitting
- Recognize overfitting vs. underfitting by comparing train/test accuracy across tree depths

## The Problem

You have a table of examples — each row is an object described by features, each with a known label (e.g., malignant/benign tumor). You want a model that looks at a **new, unlabeled** row and predicts its label correctly.

The simplest possible model a human already uses intuitively: ask a sequence of questions. "Is the cell radius large?" → if yes, ask another question; if no, ask a different one. A decision tree just formalizes and automates *which* questions to ask and *in what order*.

The hard part isn't asking questions — it's choosing the question that actually helps.

## The Concept

```
flowchart TD
    A[All training data] -->|best feature test| B[Split: condition true]
    A -->|best feature test| C[Split: condition false]
    B -->|next best feature| D[Leaf: predict label]
    B -->|next best feature| E[Leaf: predict label]
    C -->|next best feature| F[Leaf: predict label]
    C -->|next best feature| G[Leaf: predict label]
```

Three things to remember:

1. **Entropy** measures how "mixed" a set of labels is (0 = pure, 1 = maximally uncertain for binary labels)
2. **Information Gain** = entropy before split − weighted entropy after split. Pick the feature/threshold that maximizes this.
3. **max_depth is your dial between underfitting and overfitting** — shallow trees miss patterns, deep trees memorize noise.

## Build It

### Step 1: Entropy and Information Gain

```python
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))

def information_gain(y, y_left, y_right):
    n = len(y)
    weighted = (len(y_left)/n)*entropy(y_left) + (len(y_right)/n)*entropy(y_right)
    return entropy(y) - weighted
```

### Step 2: Recursive tree builder

```python
def _build(self, X, y, depth):
    if len(np.unique(y)) == 1 or depth >= self.max_depth:
        return Node(label=np.bincount(y).argmax())
    feat, thresh, gain = self._best_split(X, y)
    if feat is None or gain <= 0:
        return Node(label=np.bincount(y).argmax())
    left_mask = X[:, feat] <= thresh
    return Node(feature=feat, threshold=thresh,
                left=self._build(X[left_mask], y[left_mask], depth+1),
                right=self._build(X[~left_mask], y[~left_mask], depth+1))
```

### Step 3: Run it on real data and check for overfitting

```bash
python3 decision_tree.py
```

Expected output includes a train/test accuracy table across depths 1–20 — watch train accuracy climb to 1.0 while test accuracy plateaus or drops. That gap *is* overfitting.

## Use It

| Command / API | When |
|---|---|
| `DecisionTreeScratch(max_depth=k).fit(X, y)` | Train on your data |
| `.predict(X_test)` | Get predictions |
| `tree.root.feature` | Inspect the top-level decision (interpretability) |
| `sklearn.tree.DecisionTreeClassifier` | Production-grade version with pruning, etc. |

## Exercises

1. Change `max_depth` to `1` and `20` — explain the accuracy pattern you see in your own words.
2. Add a `min_samples_split` stopping condition and re-run the depth sweep.
3. Swap the dataset for another `sklearn.datasets` classification set (e.g. `load_wine`) and report accuracy.

## Key Terms

| Term | What people say | What it actually means |
|---|---|---|
| Entropy | "Randomness" | A number quantifying how mixed a label distribution is |
| Information Gain | "Best split" | Reduction in entropy achieved by a candidate split |
| Overfitting | "Memorizing" | Model fits training noise, fails to generalize to new data |
| Hyperparameter | "A setting" | A choice made *before* training (e.g. max_depth), not learned from data |
