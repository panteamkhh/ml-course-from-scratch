# ML-Course-From-Scratch

** from-scratch machine learning course — implemented, tested, and benchmarked against scikit-learn.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![License](https://img.shields.io/badge/license-educational-lightgrey)]()
[![Source](https://img.shields.io/badge/based%20on-A%20Course%20in%20ML-informational)]()

---

## Overview

This repository is a structured, 14-day walkthrough of *A Course in Machine Learning* by Hal Daumé III (19 chapters, 193 pages). The book is theory-only and contains no code — every algorithm here is implemented from scratch, tested on a real dataset, and cross-checked against scikit-learn's reference implementation.

Each day/chapter includes:

- A from-scratch Python implementation (no shortcuts through `model.fit()`)
- Evaluation on a real dataset (never synthetic toy data)
- A correctness check against the equivalent scikit-learn model
- A written companion guide (`en.md`) covering theory, code walkthrough, and exercises

---

## Repository Structure

```
ml-course-from-scratch/
│
├── ch01-decision-trees/
│   └── (Day 1: Decision Trees)
│
├── ch02-geometry-neighbors/
│   └── (Day 2: Geometry and Nearest Neighbors)
│
├── ch03-perceptron/
│   └── (Day 3: The Perceptron)
│
├── ch04-practical-issues-beyond-binary/
│   └── (Day 4: Practical Issues + Beyond Binary Classification)
│
├── ch05-linear-models/
│   └── (Day 5: Linear Models)
│
├── ch06-probabilistic-modeling/
│   └── (Day 6: Probabilistic Modeling)
│
├── ch07-neural-networks/
│   └── (Day 7: Neural Networks)
│
├── ch08-kernel-methods/
│   └── (Day 8: Kernel Methods)
│
├── ch09-learning-theory/
│   └── (Day 9: Learning Theory)
│
├── ch10-ensemble-efficient-learning/
│   └── (Day 10: Ensemble Methods + Efficient Learning)
│
├── ch11-unsupervised-learning/
│   └── (Day 11: Unsupervised Learning)
│
├── ch12-em-semi-supervised/
│   └── (Day 12: Expectation Maximization + Semi-Supervised Learning)
│
├── ch13-graphical-models-online-learning/
│   └── (Day 13: Graphical Models + Online Learning)
│
├── ch14-structured-bayesian-learning/
│   └── (Day 14: Structured Learning + Bayesian Learning — جمع‌بندی)
│
└── README.md
```

| File | Purpose |
|---|---|
| `*.py` | From-scratch implementation + experiments on real data |
| `en.md` | Companion write-up: learning objectives, concept, build steps, exercises |

---

## Course Schedule

| Day | Chapter(s) | Status |
|:---:|---|:---:|
| 1 | Decision Trees | Done |
| 2 | Geometry & Nearest Neighbors | Done |
| 3 | The Perceptron | Done |
| 4 | Practical Issues + Beyond Binary Classification | Pending |
| 5 | Linear Models | Pending |
| 6 | Probabilistic Modeling | Pending |
| 7 | Neural Networks | Pending |
| 8 | Kernel Methods | Pending |
| 9 | Learning Theory | Pending |
| 10 | Ensemble Methods + Efficient Learning | Pending |
| 11 | Unsupervised Learning | Pending |
| 12 | Expectation Maximization + Semi-Supervised Learning | Pending |
| 13 | Graphical Models + Online Learning | Pending |
| 14 | Structured Learning + Bayesian Learning (wrap-up) | Pending |


---

## Project Philosophy

The source book is theory-first and code-free. The goal of this repository is to translate every formula and algorithm into runnable, readable code — so that running a single script both reinforces the underlying concept and verifies correctness against an established library.

---

## Source

Hal Daumé III — *A Course in Machine Learning* — [ciml.info](http://ciml.info)

## License

This repository is intended for educational purposes only.
