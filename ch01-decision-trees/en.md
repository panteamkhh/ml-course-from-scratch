# Chapter 1: Decision Trees

> **Learning means generalizing beyond training data, not memorizing it.** The key is to distinguish between training error and test error.

**Type:** Learn + Build | **Languages:** Python | **Prerequisites:** None | **Time:** ~30 minutes  
**Source:** A Course in Machine Learning, Hal Daumé III — Chapter 1

---

## Learning Objectives

- Explain the difference between memorization and **generalization**
- Define **inductive bias** and recognize its role in learning
- Cast a concrete task as a formal learning problem (input space, features, output space, loss function)
- Illustrate how **regularization** (via `max_depth`) trades off underfitting vs overfitting
- Evaluate whether using test data is "cheating" or not

---

## The Problem

**Scenario:** You're building a course recommendation system. Given a student's past course ratings and characteristics (easy course? AI-related? morning time slot?), predict if they will like a new course.

**Challenge:** How do you learn a prediction rule from past examples that will generalize to unseen students?

**Key Insight:** If you simply memorize past examples, your learned function will fail on new data. You need to find patterns that *generalize*.

---

## The Concept

### Core Idea: Divide and Conquer

A **decision tree** repeatedly asks binary questions about features, partitioning the data until each region is "pure" (contains mostly one label).

```
                   [Is course in Systems?]
                    /                    \
                  NO                     YES
                /                          \
         [Easy?]                     [Hard-margin SVM?]
         /      \                         /
       NO       YES                      ...
      ...       ...
```

**Algorithm 1: DecisionTreeTrain** (Chapter 1, Algorithm 1.3)
1. Start with all examples and all features available
2. **Base case:** If labels are pure → return a leaf with the majority label
3. **Recursive case:** Find the feature that best splits the data (highest accuracy if we split here)
4. Partition data by that feature; remove it from future use
5. Recursively build left subtree (feature=0) and right subtree (feature=1)

**Algorithm 2: DecisionTreeTest** (Chapter 1, Algorithm 1.3)
1. Start at the root
2. If it's a leaf → return its guess
3. If it's a node → check the feature value in the test point
4. Go left (feature=0) or right (feature=1) and recurse

### Key Concepts

- **Feature:** A question you can ask (e.g., "Is this a Systems course?")
- **Label:** The correct answer (e.g., +2 for "liked", -2 for "hated")
- **Training error:** Accuracy on data the model saw
- **Test error:** Accuracy on new, unseen data ← *This is what matters*
- **Generalization:** Model does well on test data, not just train data
- **Inductive bias:** The model assumes certain patterns are "simple" and likely

---

## Build It

### Step 1: Represent the Tree

```python
class Leaf:
    def __init__(self, guess):
        self.guess = guess

class Node:
    def __init__(self, feature, left, right):
        self.feature = feature
        self.left = left
        self.right = right
```

### Step 2: Find the Best Feature (Greedy Split)

For each feature, compute: *score = # correct in left group + # correct in right group*

```python
best_feature = None
best_score = -1

for feature_idx in remaining_features:
    no_mask = (X[:, feature_idx] == 0)
    yes_mask = (X[:, feature_idx] == 1)
    
    no_y, yes_y = y[no_mask], y[yes_mask]
    
    # Majority class accuracy on each side
    score = (np.sum(no_y == mode(no_y)) + 
             np.sum(yes_y == mode(yes_y)))
    
    if score > best_score:
        best_score, best_feature = score, feature_idx
```

### Step 3: Implement the Recursive Algorithm

```python
def decision_tree_train(X, y, remaining_features, depth=0, max_depth=None):
    guess = most_frequent_label(y)
    
    # Base case: pure node
    if len(np.unique(y)) <= 1:
        return Leaf(guess)
    
    # Base case: no features left
    if len(remaining_features) == 0:
        return Leaf(guess)
    
    # Base case: hit max depth (REGULARIZATION — see Section 1.9)
    if max_depth is not None and depth >= max_depth:
        return Leaf(guess)
    
    # Find best split (greedy)
    best_feature = find_best_feature(X, y, remaining_features)
    
    if best_feature is None:
        return Leaf(guess)
    
    # Partition and recurse
    no_mask = (X[:, best_feature] == 0)
    yes_mask = (X[:, best_feature] == 1)
    
    left = decision_tree_train(X[no_mask], y[no_mask], 
                                remaining_features - {best_feature},
                                depth + 1, max_depth)
    right = decision_tree_train(X[yes_mask], y[yes_mask],
                                remaining_features - {best_feature},
                                depth + 1, max_depth)
    
    return Node(best_feature, left, right)
```

### Step 4: Prediction

```python
def decision_tree_predict_single(tree, x):
    if isinstance(tree, Leaf):
        return tree.guess
    
    if x[tree.feature] == 1:
        return decision_tree_predict_single(tree.right, x)
    else:
        return decision_tree_predict_single(tree.left, x)
```

### Step 5: Run on Real Data

```bash
$ python3 decision_tree.py
```

**Output:**
```
max_depth |  train_acc |   test_acc
----------------------------------
         1 |     0.8615 |     0.7632
         2 |     0.9209 |     0.8772
         3 |     0.9275 |     0.9298  ← sweet spot
         4 |     0.9429 |     0.9123
       ... |        ... |        ...
      None |     0.9978 |     0.9386  ← overfitting
```

**Key observation:**
- **max_depth=1**: Underfitting (tree is too shallow)
- **max_depth=3**: Good generalization
- **max_depth=None**: Overfitting (tree memorizes noise)

---

## Use It

### When to Use Decision Trees

✓ Interpretability is critical  
✓ Mixed feature types (continuous, categorical)  
✓ Many irrelevant features  
✓ Speed is critical at test time  
✗ Very large datasets (training is slow)  
✗ XOR problem (non-linear patterns)  

---

## Key Terms

| Term | Meaning |
|---|---|
| **Generalization** | Test error ≈ train error |
| **Overfitting** | Train error << test error (memorized noise) |
| **Underfitting** | Both train and test error high (too simple) |
| **Regularization** | Constraint that prevents overfitting (e.g., `max_depth`) |
| **Hyperparameter** | Parameter set before training (e.g., `max_depth`) |

---

## Summary

✓ Decision trees learn by greedily partitioning data using binary questions  
✓ Generalization (test error) is the goal, not memorization (train error)  
✓ `max_depth` is a regularizer that controls underfitting/overfitting  
✓ Test accuracy on Breast Cancer Wisconsin: **0.9386**  
✓ Trees are interpretable and fast, but can overfit  

---

**Next:** Chapter 2: Geometry and Nearest Neighbors
