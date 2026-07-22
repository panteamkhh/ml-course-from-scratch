"""
Chapter 18: Structured Learning Tasks
A Course in Machine Learning (Hal Daume III)

The book chapter itself is only an outline (HMMs, structured perceptron,
CRFs, M3Ns). This script implements the core idea from scratch:

    structured perceptron = perceptron (Ch. 3) + Viterbi decoding

applied to part-of-speech (POS) tagging, a classic sequence-labeling
("structured") task: the output is not a single label but a *sequence*
of labels, and neighboring labels are correlated (e.g. a DET is almost
always followed by a NOUN or ADJ, never by another DET).

We compare two models on the SAME real, hand-tagged English sentences:

  1. Baseline: an independent per-token linear classifier (like OVA
     from Chapter 5) that ignores structure completely.
  2. Structured perceptron: learns both emission weights (word/context
     -> tag) AND transition weights (tag -> tag), and decodes whole
     sentences at once with the Viterbi algorithm.

This isolates exactly the value that "structure" adds over treating
each token as an independent classification problem.
"""

import numpy as np
from collections import defaultdict
import random

random.seed(0)
np.random.seed(0)

# --------------------------------------------------------------------------
# REAL DATA: hand-tagged English sentences (Penn-Treebank-style tagset,
# collapsed to a small universal set: NOUN, VERB, ADJ, DET, ADP, PRON,
# ADV, CONJ, NUM, PUNC). These are genuine English sentences with
# linguistically correct tags -- not synthetic/random data.
# --------------------------------------------------------------------------
TAGGED_SENTENCES = [
    "The/DET dog/NOUN chased/VERB the/DET cat/NOUN ./PUNC",
    "A/DET quick/ADJ brown/ADJ fox/NOUN jumps/VERB over/ADP the/DET lazy/ADJ dog/NOUN ./PUNC",
    "She/PRON sells/VERB seashells/NOUN by/ADP the/DET seashore/NOUN ./PUNC",
    "He/PRON gave/VERB her/PRON a/DET beautiful/ADJ gift/NOUN ./PUNC",
    "The/DET students/NOUN studied/VERB machine/NOUN learning/NOUN today/ADV ./PUNC",
    "Two/NUM cats/NOUN slept/VERB on/ADP the/DET warm/ADJ couch/NOUN ./PUNC",
    "I/PRON really/ADV enjoy/VERB reading/VERB good/ADJ books/NOUN ./PUNC",
    "The/DET teacher/NOUN explained/VERB the/DET difficult/ADJ theorem/NOUN clearly/ADV ./PUNC",
    "My/PRON sister/NOUN and/CONJ I/PRON walked/VERB to/ADP the/DET store/NOUN ./PUNC",
    "The/DET old/ADJ car/NOUN broke/VERB down/ADP suddenly/ADV ./PUNC",
    "They/PRON quickly/ADV finished/VERB their/PRON homework/NOUN ./PUNC",
    "A/DET tall/ADJ man/NOUN opened/VERB the/DET heavy/ADJ door/NOUN ./PUNC",
    "The/DET children/NOUN played/VERB happily/ADV in/ADP the/DET park/NOUN ./PUNC",
    "Three/NUM birds/NOUN flew/VERB over/ADP the/DET tall/ADJ tree/NOUN ./PUNC",
    "She/PRON carefully/ADV painted/VERB the/DET small/ADJ wooden/ADJ box/NOUN ./PUNC",
    "The/DET scientist/NOUN discovered/VERB a/DET new/ADJ planet/NOUN yesterday/ADV ./PUNC",
    "He/PRON and/CONJ his/PRON friend/NOUN built/VERB a/DET small/ADJ boat/NOUN ./PUNC",
    "The/DET bright/ADJ sun/NOUN warmed/VERB the/DET cold/ADJ ground/NOUN ./PUNC",
    "We/PRON watched/VERB the/DET beautiful/ADJ sunset/NOUN together/ADV ./PUNC",
    "The/DET young/ADJ girl/NOUN sang/VERB a/DET sweet/ADJ song/NOUN ./PUNC",
    "Four/NUM dogs/NOUN barked/VERB loudly/ADV at/ADP the/DET stranger/NOUN ./PUNC",
    "The/DET chef/NOUN prepared/VERB a/DET delicious/ADJ meal/NOUN quickly/ADV ./PUNC",
    "My/PRON brother/NOUN drives/VERB an/DET old/ADJ blue/ADJ truck/NOUN ./PUNC",
    "The/DET river/NOUN flows/VERB gently/ADV through/ADP the/DET valley/NOUN ./PUNC",
    "A/DET small/ADJ bird/NOUN built/VERB its/PRON nest/NOUN carefully/ADV ./PUNC",
    "The/DET manager/NOUN reviewed/VERB the/DET final/ADJ report/NOUN twice/ADV ./PUNC",
    "They/PRON planted/VERB new/ADJ flowers/NOUN in/ADP the/DET garden/NOUN ./PUNC",
    "The/DET tired/ADJ hiker/NOUN rested/VERB near/ADP the/DET quiet/ADJ lake/NOUN ./PUNC",
    "She/PRON quickly/ADV solved/VERB the/DET hard/ADJ puzzle/NOUN ./PUNC",
    "The/DET committee/NOUN approved/VERB the/DET new/ADJ policy/NOUN today/ADV ./PUNC",
]


def parse(sentences):
    data = []
    for s in sentences:
        words, tags = [], []
        for tok in s.split():
            w, t = tok.rsplit("/", 1)
            words.append(w.lower())
            tags.append(t)
        data.append((words, tags))
    return data


DATA = parse(TAGGED_SENTENCES)
random.shuffle(DATA)
split = int(0.8 * len(DATA))
TRAIN, TEST = DATA[:split], DATA[split:]

TAGS = sorted({t for _, tags in DATA for t in tags})
TAG2I = {t: i for i, t in enumerate(TAGS)}
K = len(TAGS)


# --------------------------------------------------------------------------
# Feature extraction: emission features for word i in a sentence.
# (This mirrors the "features = questions you can ask" view from Ch. 1/2.)
# --------------------------------------------------------------------------
def emission_features(words, i):
    w = words[i]
    feats = [
        f"word={w}",
        f"suffix2={w[-2:]}",
        f"suffix3={w[-3:]}",
        f"is_first={i == 0}",
        f"is_last={i == len(words) - 1}",
        f"is_punct={w in '.,!?;'}",
    ]
    return feats


# ==========================================================================
# Model 1: Baseline -- independent per-token linear classifier (multiclass
# perceptron / one-vs-all in the style of Chapter 5), no structure at all.
# ==========================================================================
class IndependentTokenPerceptron:
    def __init__(self):
        self.w = defaultdict(lambda: np.zeros(K))

    def score(self, feats):
        s = np.zeros(K)
        for f in feats:
            s += self.w[f]
        return s

    def fit(self, data, epochs=15):
        for _ in range(epochs):
            for words, tags in data:
                for i in range(len(words)):
                    feats = emission_features(words, i)
                    y = TAG2I[tags[i]]
                    scores = self.score(feats)
                    yhat = int(np.argmax(scores))
                    if yhat != y:
                        for f in feats:
                            self.w[f][y] += 1.0
                            self.w[f][yhat] -= 1.0

    def predict(self, words):
        return [TAGS[int(np.argmax(self.score(emission_features(words, i))))]
                for i in range(len(words))]


# ==========================================================================
# Model 2: Structured Perceptron with Viterbi decoding.
#   score(words, tags) = sum_i emission(words, i, tags[i])
#                       + sum_i transition(tags[i-1], tags[i])
# Training follows Algorithm 5 (perceptron) generalized to structured
# outputs: predict the best full sequence, and if wrong, bump up the
# weights of the true sequence's features and bump down the predicted
# sequence's features (the structured analogue of "wd <- wd + y*xd").
# ==========================================================================
class StructuredPerceptron:
    def __init__(self):
        self.emit_w = defaultdict(lambda: np.zeros(K))
        self.trans_w = np.zeros((K + 1, K))  # +1 for BOS "previous tag"
        self.BOS = K

    def emit_scores(self, words, i):
        s = np.zeros(K)
        for f in emission_features(words, i):
            s += self.emit_w[f]
        return s

    def viterbi(self, words):
        N = len(words)
        V = np.full((N, K), -np.inf)
        back = np.zeros((N, K), dtype=int)
        e0 = self.emit_scores(words, 0)
        V[0] = e0 + self.trans_w[self.BOS]
        for i in range(1, N):
            ei = self.emit_scores(words, i)
            # V[i][k] = max_prev V[i-1][prev] + trans[prev][k] + emit[k]
            candidates = V[i - 1][:, None] + self.trans_w[:K, :]  # (K,K)
            back[i] = np.argmax(candidates, axis=0)
            V[i] = np.max(candidates, axis=0) + ei
        # backtrack
        tags_idx = [int(np.argmax(V[N - 1]))]
        for i in range(N - 1, 0, -1):
            tags_idx.append(int(back[i, tags_idx[-1]]))
        tags_idx.reverse()
        return [TAGS[t] for t in tags_idx]

    def _update(self, words, gold, pred, lr=1.0):
        prev_gold, prev_pred = self.BOS, self.BOS
        for i in range(len(words)):
            gy, py = TAG2I[gold[i]], TAG2I[pred[i]]
            if gy != py or prev_gold != prev_pred:
                for f in emission_features(words, i):
                    self.emit_w[f][gy] += lr
                    self.emit_w[f][py] -= lr
                self.trans_w[prev_gold, gy] += lr
                self.trans_w[prev_pred, py] -= lr
            prev_gold, prev_pred = gy, py

    def fit(self, data, epochs=15):
        for _ in range(epochs):
            random.shuffle(data)
            for words, tags in data:
                pred = self.viterbi(words)
                if pred != tags:
                    self._update(words, tags, pred)

    def predict(self, words):
        return self.viterbi(words)


def token_accuracy(model, data):
    correct, total = 0, 0
    for words, tags in data:
        pred = model.predict(words)
        correct += sum(p == g for p, g in zip(pred, tags))
        total += len(tags)
    return correct / total


def sentence_accuracy(model, data):
    correct = 0
    for words, tags in data:
        if model.predict(words) == tags:
            correct += 1
    return correct / len(data)


if __name__ == "__main__":
    print("=" * 70)
    print("CHAPTER 18: STRUCTURED LEARNING -- POS Tagging")
    print("=" * 70)
    print(f"Sentences: {len(DATA)} total ({len(TRAIN)} train / {len(TEST)} test)")
    print(f"Tagset ({K} tags): {TAGS}")
    print()

    baseline = IndependentTokenPerceptron()
    baseline.fit(TRAIN, epochs=25)

    structured = StructuredPerceptron()
    structured.fit(TRAIN, epochs=25)

    print("-" * 70)
    print("RESULTS (held-out test sentences)")
    print("-" * 70)
    print(f"{'Model':<32} | {'token acc':>9} | {'sentence acc':>12}")
    print("-" * 70)
    for name, model in [("Independent per-token (no structure)", baseline),
                         ("Structured perceptron (+ Viterbi)", structured)]:
        ta = token_accuracy(model, TEST)
        sa = sentence_accuracy(model, TEST)
        print(f"{name:<32} | {ta:>9.4f} | {sa:>12.4f}")

    print()
    print("Example sentence (unseen structure of tags):")
    words, gold = TEST[0]
    print("  words :", " ".join(words))
    print("  gold  :", " ".join(gold))
    print("  base  :", " ".join(baseline.predict(words)))
    print("  struct:", " ".join(structured.predict(words)))

    print()
    print("Learned transition preferences (top-3 'what comes after DET'):")
    det_row = structured.trans_w[TAG2I["DET"]]
    order = np.argsort(-det_row)[:3]
    for idx in order:
        print(f"  DET -> {TAGS[idx]:<6} weight={det_row[idx]:.2f}")
