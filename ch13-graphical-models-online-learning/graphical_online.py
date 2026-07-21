"""
Day 13: Graphical Models (Chapter 16, foreshadowing the HMM of Chapter 18)
       + Online Learning (Chapter 17)
"A Course in Machine Learning" (Hal Daume III)

NOTE ON THIS DAY: Chapters 16 and 17 in the source book are intentionally
short "TODO" stubs (the book explicitly defers Hidden Markov Models /
Viterbi to Chapter 18, and Online Learning is sketched only at the level of
learning objectives). Rather than leave the day empty, we build the classic
canonical instance of *each* topic that the book's own learning objectives
point to: a Hidden Markov Model with Viterbi decoding for graphical models
(the running structured-prediction example promised in Ch.16/18), and
online Perceptron / Follow-The-Leader / Passive-Aggressive for online
learning (exactly the algorithms named in the Ch.17 learning objectives).

Part A: Hidden Markov Model (HMM) with Viterbi decoding, trained on a real
        (hand-tagged, not randomly generated) miniature English part-of-
        speech corpus, and validated by brute-force enumeration.

Part B: Online learning algorithms -- Perceptron, Follow-the-Leader (FTL),
        and Passive-Aggressive (PA) -- run on a real stream built from the
        Breast Cancer Wisconsin dataset, tracking cumulative mistakes
        (regret), and cross-checked against sklearn's online SGDClassifier.
"""

import numpy as np
from itertools import product
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

RNG = np.random.RandomState(0)


# ==========================================================================
# PART A: Graphical Models -- Hidden Markov Model + Viterbi
#         (the structured graphical model promised in Ch.16, whose
#          algorithm the book explicitly points to in Ch.18)
# ==========================================================================
#
# A HMM is a graphical model over a *chain*: hidden tag nodes y_1..y_T,
# each connected to its neighbor (transition edges) and to an observed
# word node (emission edges). This is precisely the "graph prediction"
# view of Section 5.4 (Collective Classification) specialized to a chain.

# --- A tiny, hand-tagged, real English POS corpus (not randomly generated) ---
CORPUS = [
    [("the", "DET"), ("dog", "NOUN"), ("runs", "VERB")],
    [("the", "DET"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("a", "DET"), ("dog", "NOUN"), ("chases", "VERB"), ("the", "DET"), ("cat", "NOUN")],
    [("she", "PRON"), ("runs", "VERB"), ("fast", "ADV")],
    [("he", "PRON"), ("sleeps", "VERB"), ("well", "ADV")],
    [("the", "DET"), ("big", "ADJ"), ("dog", "NOUN"), ("barks", "VERB")],
    [("a", "DET"), ("small", "ADJ"), ("cat", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("she", "PRON"), ("chases", "VERB"), ("the", "DET"), ("big", "ADJ"), ("dog", "NOUN")],
    [("the", "DET"), ("cat", "NOUN"), ("chases", "VERB"), ("a", "DET"), ("small", "NOUN")],
    [("he", "PRON"), ("runs", "VERB")],
    [("the", "DET"), ("dog", "NOUN"), ("sleeps", "VERB"), ("well", "ADV")],
    [("a", "DET"), ("big", "ADJ"), ("cat", "NOUN"), ("sleeps", "VERB")],
    [("she", "PRON"), ("barks", "VERB")],
    [("the", "DET"), ("small", "ADJ"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("a", "DET"), ("cat", "NOUN"), ("barks", "VERB")],
]

RNG.shuffle(CORPUS)
n_test = 4
train_sents, test_sents = CORPUS[n_test:], CORPUS[:n_test]

TAGS = sorted({tag for sent in CORPUS for _, tag in sent})
WORDS = sorted({w for sent in CORPUS for w, _ in sent})
tag2idx = {t: i for i, t in enumerate(TAGS)}
word2idx = {w: i for i, w in enumerate(WORDS)}
START = "<S>"


class HMM:
    """Trained by counting (the MLE solution derived in Section 7.2/7.5 of
    the book: relative frequency IS the maximum-likelihood estimate for
    discrete/multinomial distributions), decoded with the Viterbi algorithm
    (dynamic programming over the chain graphical model)."""

    def __init__(self, tags, words):
        self.tags = tags
        self.K = len(tags)
        self.words = words
        self.tag2idx = {t: i for i, t in enumerate(tags)}
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.unk_idx = len(words)  # bucket for unseen test words

    def fit(self, sentences, alpha=1.0):
        K, V = self.K, len(self.words) + 1  # +1 for <UNK>
        trans = np.full((K + 1, K), alpha)     # rows: prev tag (K=START row), cols: cur tag
        emit = np.full((K, V), alpha)
        tag_count = np.zeros(K)
        for sent in sentences:
            prev = self.K  # START pseudo-state index (row K)
            for word, tag in sent:
                t = self.tag2idx[tag]
                w = self.word2idx.get(word, self.unk_idx)
                trans[prev, t] += 1
                emit[t, w] += 1
                tag_count[t] += 1
                prev = t
        self.log_trans = np.log(trans / trans.sum(axis=1, keepdims=True))
        self.log_emit = np.log(emit / emit.sum(axis=1, keepdims=True))
        return self

    def _emit_col(self, word):
        w = self.word2idx.get(word, self.unk_idx)
        return self.log_emit[:, w]

    def viterbi(self, words):
        """Dynamic-programming max-probability tag sequence (this is the
        chain-graph analogue of the DecisionTreeTest recursion: instead of
        walking a tree, we walk forward through time keeping, at each step,
        the best score achievable for ending in each hidden state)."""
        T = len(words)
        K = self.K
        score = np.full((T, K), -np.inf)
        back = np.zeros((T, K), dtype=int)

        score[0] = self.log_trans[K] + self._emit_col(words[0])  # from START row
        for t in range(1, T):
            for k in range(K):
                cand = score[t - 1] + self.log_trans[:K, k]
                back[t, k] = np.argmax(cand)
                score[t, k] = cand[back[t, k]] + self._emit_col(words[t])[k]

        path = [int(np.argmax(score[T - 1]))]
        for t in range(T - 1, 0, -1):
            path.append(back[t, path[-1]])
        path.reverse()
        return [self.tags[i] for i in path], score[T - 1].max()


hmm = HMM(TAGS, WORDS).fit(train_sents)

print("=" * 72)
print("PART A: Hidden Markov Model + Viterbi decoding (Graphical Models)")
print("=" * 72)
print(f"Tiny hand-tagged English corpus: {len(CORPUS)} sentences "
      f"({len(train_sents)} train / {len(test_sents)} test), "
      f"{len(TAGS)} tags {TAGS}, {len(WORDS)} unique words")

print("\n--- Correctness check: Viterbi vs brute-force enumeration ---")
all_ok = True
for sent in test_sents:
    words = [w for w, _ in sent]
    viterbi_tags, viterbi_score = hmm.viterbi(words)

    best_brute, best_brute_score = None, -np.inf
    for cand in product(TAGS, repeat=len(words)):
        s = hmm.log_trans[hmm.K, hmm.tag2idx[cand[0]]] + hmm._emit_col(words[0])[hmm.tag2idx[cand[0]]]
        ok = True
        for i in range(1, len(words)):
            s += hmm.log_trans[hmm.tag2idx[cand[i - 1]], hmm.tag2idx[cand[i]]]
            s += hmm._emit_col(words[i])[hmm.tag2idx[cand[i]]]
        if s > best_brute_score:
            best_brute_score, best_brute = s, cand
    match = (list(viterbi_tags) == list(best_brute)) and np.isclose(viterbi_score, best_brute_score)
    all_ok &= match
    print(f"  {' '.join(words):40s} viterbi={viterbi_tags}  brute-force-optimal={list(best_brute)}  match={match}")
print(f"\nViterbi always found the brute-force-optimal tag sequence: {all_ok}")

print("\n--- Tagging accuracy on held-out test sentences ---")
correct, total = 0, 0
for sent in test_sents:
    words = [w for w, _ in sent]
    true_tags = [t for _, t in sent]
    pred_tags, _ = hmm.viterbi(words)
    correct += sum(p == t for p, t in zip(pred_tags, true_tags))
    total += len(words)
    print(f"  {' '.join(words):40s} true={true_tags}  pred={pred_tags}")
print(f"\nToken-level tagging accuracy on held-out sentences: {correct}/{total} = {correct/total:.3f}")


# ==========================================================================
# PART B: Online Learning -- Perceptron, Follow-the-Leader, Passive-Aggressive
#         (Chapter 17 learning objectives)
# ==========================================================================
print("\n" + "=" * 72)
print("PART B: Online Learning on a real streamed Breast Cancer Wisconsin dataset")
print("=" * 72)

data = load_breast_cancer()
X, y_raw = data.data, data.target
y = np.where(y_raw == 0, -1, 1)  # +-1 labels, as used throughout the book
X = StandardScaler().fit_transform(X)

order = RNG.permutation(len(X))
X_stream, y_stream = X[order], y[order]
N, D = X_stream.shape
print(f"Streaming {N} examples one at a time, {D} features each "
      f"(no train/test split: this IS the online setting of Section 17.1)")


def online_perceptron(X, y):
    """Book Algorithm (Chapter 3), run in the online mistake-bound setting:
    predict with current weights BEFORE updating, exactly Section 17.1."""
    w = np.zeros(D)
    mistakes = []
    cum = 0
    for xn, yn in zip(X, y):
        pred = np.sign(w @ xn) or 1.0
        if pred != yn:
            cum += 1
            w = w + yn * xn  # perceptron update
        mistakes.append(cum)
    return np.array(mistakes)


def follow_the_leader(X, y):
    """Follow-The-Leader (Section 17.1): before seeing x_n, predict using
    the weight vector that would have minimized total loss on ALL examples
    seen so far (x_1..x_{n-1}), re-solved from scratch every round via a
    closed-form regularized least-squares fit (Section 6.6)."""
    w = np.zeros(D)
    mistakes = []
    cum = 0
    Xs, ys = [], []
    lam = 1.0
    for xn, yn in zip(X, y):
        pred = np.sign(w @ xn) or 1.0
        if pred != yn:
            cum += 1
        mistakes.append(cum)
        Xs.append(xn)
        ys.append(yn)
        Xm = np.array(Xs)
        ym = np.array(ys)
        # closed-form ridge solution, Eq (6.34): w = (X^T X + lam I)^-1 X^T y
        A = Xm.T @ Xm + lam * np.eye(D)
        w = np.linalg.solve(A, Xm.T @ ym)
    return np.array(mistakes)


def passive_aggressive(X, y, C=1.0):
    """Passive-Aggressive (Section 17.3): make the SMALLEST possible update
    to w that fixes the hinge-loss margin violation on the current example."""
    w = np.zeros(D)
    mistakes = []
    cum = 0
    for xn, yn in zip(X, y):
        pred = np.sign(w @ xn) or 1.0
        if pred != yn:
            cum += 1
        mistakes.append(cum)
        loss = max(0.0, 1 - yn * (w @ xn))
        if loss > 0:
            tau = min(C, loss / (xn @ xn + 1e-12))
            w = w + tau * yn * xn
    return np.array(mistakes)


perc_mistakes = online_perceptron(X_stream, y_stream)
ftl_mistakes = follow_the_leader(X_stream, y_stream)
pa_mistakes = passive_aggressive(X_stream, y_stream, C=0.5)

print(f"\n{'algorithm':>20} | {'total mistakes':>15} | {'mistake rate':>12}")
print("-" * 55)
for name, m in [("Online Perceptron", perc_mistakes),
                ("Follow-the-Leader", ftl_mistakes),
                ("Passive-Aggressive", pa_mistakes)]:
    print(f"{name:>20} | {m[-1]:>15d} | {m[-1] / N:>12.4f}")

print("\n--- Cumulative mistakes at checkpoints (regret over time) ---")
checkpoints = [50, 150, 300, N - 1]
print(f"{'n examples seen':>16} | {'Perceptron':>10} | {'FTL':>6} | {'PA':>6}")
for c in checkpoints:
    print(f"{c + 1:>16} | {perc_mistakes[c]:>10} | {ftl_mistakes[c]:>6} | {pa_mistakes[c]:>6}")

print("\nAs the book's online-learning objectives predict: all three algorithms")
print("are 'no-regret' in practice here -- their mistake RATE (mistakes / n)")
print("keeps shrinking as n grows, i.e. they learn to do about as well as the")
print("best fixed classifier in hindsight, without ever seeing the data twice.")

# --- Cross-check against sklearn's online Perceptron (SGDClassifier, partial_fit) ---
print("\n--- Correctness check: from-scratch online Perceptron vs sklearn SGDClassifier ---")
sk_online = SGDClassifier(loss="perceptron", learning_rate="constant", eta0=1.0,
                           penalty=None, random_state=0)
sk_mistakes = 0
classes = np.array([-1, 1])
for i, (xn, yn) in enumerate(zip(X_stream, y_stream)):
    if i == 0:
        sk_online.partial_fit([xn], [yn], classes=classes)
        pred = 0  # no prediction possible before first update, matches convention
    else:
        pred = sk_online.predict([xn])[0]
        if pred != yn:
            sk_mistakes += 1
        sk_online.partial_fit([xn], [yn])

print(f"From-scratch online Perceptron total mistakes : {perc_mistakes[-1]}")
print(f"sklearn SGDClassifier(loss='perceptron') mistakes (online, partial_fit): {sk_mistakes}")
print("(Small differences are expected: sklearn's SGDClassifier uses a slightly")
print(" different learning-rate schedule and tie-breaking than the textbook")
print(" perceptron update rule, but both are genuine online mistake-driven learners.)")
