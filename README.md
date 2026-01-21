# Relation Extraction / Utterance Classification (PyTorch MLP)

This project is a supervised NLP classification assignment: given a natural-language **utterance** about a movie (or actor), predict the correct **Freebase-style relation(s)** the utterance invokes (e.g., `movie.initial_release_date`). The dataset focuses on the **film schema** and is framed as **multi-label classification**, since some utterances can invoke multiple relations. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Task Framing](#task-framing)
- [Model](#model)
- [Text Representation](#text-representation)
- [Training Setup](#training-setup)
- [Experiments](#experiments)
- [Results Summary](#results-summary)
- [Key Takeaways](#key-takeaways)
- [Common Pitfalls / Notes to Future Me](#common-pitfalls--notes-to-future-me)
- [How to Reproduce](#how-to-reproduce)
- [Next Improvements](#next-improvements)

---

## Project Goal
Train a deep learning model to predict the relation(s) invoked by an utterance. Example: an utterance asking about release date should map to `movies.initial_release_date`. :contentReference[oaicite:3]{index=3}

---

## Dataset
Two CSV files are used:
- `hw1_train.csv`: contains **ID**, **UTTERANCES**, **CORE RELATIONS** (2313 rows) :contentReference[oaicite:4]{index=4}
- `hw1_test.csv`: contains **ID** and **UTTERANCES** (982 rows) :contentReference[oaicite:5]{index=5}

There are **19 unique relations** in this dataset (including `none`). :contentReference[oaicite:6]{index=6}

### Relations (high level)
Relations include items like:
- `movie.initial_release_date`, `movie.directed_by`, `movie.genre`, `movie.starring.actor`, `movie.starring.character`
- plus `none` and `person.date_of_birth`, etc. :contentReference[oaicite:7]{index=7}

---

## Task Framing
This is supervised learning: inputs are utterances, outputs are relation labels. :contentReference[oaicite:8]{index=8}

### Multi-label vs multi-class
Some utterances may require multiple relations (example: a query involving actor + gender). Therefore the task is treated as **multi-label classification** rather than strictly multi-class. :contentReference[oaicite:9]{index=9}

**Practical implication**: the model should be able to output:
- zero relations (e.g., `none`)
- one relation
- multiple relations

---

## Model
The model is an **MLP (Multilayer Perceptron)** with:
- input layer (vectorized text features)
- hidden layer 1: **512** units
- hidden layer 2: **256** units
- dropout: **0.3**
- activation: **ReLU** by default, with experiments using **Leaky ReLU** :contentReference[oaicite:10]{index=10} :contentReference[oaicite:11]{index=11}

Why MLP here:
- When using sparse bag-of-words style features, an MLP is a reasonable baseline that can learn non-linear decision boundaries while remaining relatively simple.

---

## Text Representation
Text is vectorized using **CountVectorizer** (word-level tokenization by default). :contentReference[oaicite:12]{index=12}

Key vectorizer settings discussed:
- `stop_words='english'` to remove common words
- `max_features=3000` to limit vocabulary size and reduce noise :contentReference[oaicite:13]{index=13}

A major experiment later switches the analyzer to:
- `char_wb` (character-level tokenization), which produced the largest improvement. :contentReference[oaicite:14]{index=14}

**Why char-level helps (intuition):**
- Smaller, denser feature space
- Better robustness to spelling variations / morphology
- Particularly helpful on smaller datasets :contentReference[oaicite:15]{index=15}

---

## Training Setup
### Split
Training data is split:
- 80% train / 20% test :contentReference[oaicite:16]{index=16}

### Loss
Default loss: **BCEWithLogitsLoss**, appropriate for multi-label classification. :contentReference[oaicite:17]{index=17}

Ablation: **CrossEntropyLoss** is tested and found to perform poorly for this multi-label framing. The report explicitly notes CrossEntropy is more appropriate for multi-class (softmax) while BCEWithLogitsLoss is better for multi-label (sigmoid). :contentReference[oaicite:18]{index=18}

### Optimizers
Optimizers explored:
- Adam (default)
- AdamW
- NAdam
- Adagrad :contentReference[oaicite:19]{index=19}

### Hyperparameters explored
- learning rate: 0.01 vs 0.001 :contentReference[oaicite:20]{index=20}
- epochs: 100 vs 1000 :contentReference[oaicite:21]{index=21}
- activation: ReLU vs Leaky ReLU :contentReference[oaicite:22]{index=22}
- vectorizer analyzer: word vs char_wb :contentReference[oaicite:23]{index=23}

### Evaluation
The report focuses on:
- accuracy over epochs
- loss over epochs
- overall prediction accuracy :contentReference[oaicite:24]{index=24}

---

## Experiments
Experiments are structured by changing one or more hyperparameters:
- Lower learning rate improved stability vs high learning rate (less oscillation in accuracy/loss). :contentReference[oaicite:25]{index=25}
- CrossEntropy performed poorly compared to BCEWithLogitsLoss in this multi-label setup. :contentReference[oaicite:26]{index=26} :contentReference[oaicite:27]{index=27}
- Switching CountVectorizer to `char_wb` showed the biggest jump in accuracy. :contentReference[oaicite:28]{index=28}

---

## Results Summary
Best-performing configuration reported:
- optimizer: **AdamW**
- epochs: **100**
- learning rate: **0.001**
- loss: **BCEWithLogitsLoss**
- activation: **Leaky ReLU**
- analyzer: **char_wb**
- prediction accuracy: **73.4%** :contentReference[oaicite:29]{index=29}

The report notes:
- Increasing epochs to 1000 sometimes reduced prediction accuracy (possible overfitting), and graphs showed more oscillation. :contentReference[oaicite:30]{index=30} :contentReference[oaicite:31]{index=31}

A summary table of configurations and accuracies is included in the appendix. :contentReference[oaicite:32]{index=32}

---

## Key Takeaways
1. **Low learning rate matters** (reduced oscillation, improved training dynamics). :contentReference[oaicite:33]{index=33}  
2. **BCEWithLogitsLoss fits multi-label** better than CrossEntropy in this formulation. :contentReference[oaicite:34]{index=34}  
3. **Character-level tokenization (`char_wb`) was the strongest lever** for performance improvement on this dataset. :contentReference[oaicite:35]{index=35}  
4. More epochs is not always better; it can introduce oscillation and possible overfitting. :contentReference[oaicite:36]{index=36}  

---

## Common Pitfalls / Notes to Future Me
- If framing as multi-label, ensure:
  - sigmoid-style outputs (or logits + BCEWithLogitsLoss)
  - thresholding strategy at inference time (how logits become labels)
- If accuracy plateaus:
  - check learning rate first (0.001 was much more stable than 0.01)
- Tokenization choice can dominate everything:
  - word-level can miss signals if vocabulary is sparse or text varies
  - char-level can generalize better in small-data settings

---

## How to Reproduce
1. Place `hw1_train.csv` and `hw1_test.csv` in the project directory.
2. Run training using the MLP setup described:
   - Vectorize text (CountVectorizer, try both `word` and `char_wb`)
   - Train MLP with BCEWithLogitsLoss
3. Generate predictions for `hw1_test.csv` and submit (if the assignment uses Kaggle submissions).

---

## Next Improvements
If revisiting, here are high-impact additions:
- **Better evaluation for multi-label**:
  - per-label precision/recall/F1
  - micro/macro averages
- **Threshold tuning**:
  - choose thresholds per label (not necessarily 0.5)
- **Try TF-IDF** instead of raw counts
- **Try a linear baseline** (LogReg / linear SVM) with char n-grams as a sanity check
- **Use validation set / cross-validation** for more reliable comparisons
- **Regularization sweeps**:
  - dropout rate
  - weight decay (esp. with AdamW)

---
