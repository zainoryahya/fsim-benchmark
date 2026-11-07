# Dorothea: additional validation, declared before any run (2026-10-24 18:31 +08)

Status: post-hoc addition (not part of the pre-registered 18-set held-out suite or H1-H6).
One dataset only, so results are descriptive (no across-dataset test).

Data: NIPS 2003 feature selection challenge (Guyon 2003), train + valid merged (test labels withheld):
n = 1150, positives = 112 (minority, label +1), IR = 9.27, 100000 sparse binary features
(about half are artificial probes per the challenge design).
Label-free cleaning only: keep features active in >= 1% of samples (>= 12 of 1150): 32220 features.
Protocol, seeds, methods, caps, classifiers: identical to the held-out chain (chain.sh).

Predictions (from the held-out results):
P1 leakage: within-fold null MCC in [-0.03, +0.03]; SMOTE-before-CV and filter-before-CV null MCC > +0.10.
P2 SMOTE vs prior threshold: |G-mean difference| <= 0.02.
P3 wrapper cap20 (5-NN search) vs Hellinger top-20, averaged over LR and SVM: difference < +0.02.
P4 EMT-min3 vs cap20 (5-NN): difference <= +0.02 (no EMT advantage).
P5 Nogueira stability: wrapper (cap20, EMT) < Hellinger filter.
