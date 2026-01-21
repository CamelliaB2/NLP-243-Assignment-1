# Relation Extraction from Natural Language (PyTorch)

This project explores **relation extraction** as a supervised NLP task using deep learning in PyTorch. The goal is to classify which **semantic relations** are invoked by a natural-language utterance about films or people, based on a predefined schema derived from Freebase.

This repository serves as both an implementation and an experiment log, documenting how architectural and hyperparameter choices affect performance.

---

## Table of Contents
1. [Problem Overview](#problem-overview)
2. [Dataset Description](#dataset-description)
3. [Task Formulation](#task-formulation)
4. [Model Architecture](#model-architecture)
5. [Feature Representation](#feature-representation)
6. [Training Setup](#training-setup)
7. [Experiments and Findings](#experiments-and-findings)
8. [Key Lessons Learned](#key-lessons-learned)
9. [Limitations](#limitations)
10. [Possible Improvements](#possible-improvements)
11. [References](#references)

---

## Problem Overview

The task is **relation extraction** from short natural-language queries. Given an utterance (e.g., a question about a movie), the model must predict which **relation(s)** are being requested.

Example:
- Utterance: *"Who directed Inception?"*
- Target relation: `movie.directed_by`

This is a **supervised learning** problem where:
- Input: text utterance
- Output: one or more relations

---

## Dataset Description

The dataset is derived from the **Freebase film schema** and contains movie- and person-related queries.

### Training Set
- 2,313 utterances
- Columns:
  - `ID`
  - `UTTERANCE`
  - `CORE_RELATIONS`

### Test Set
- 982 utterances
- Columns:
  - `ID`
  - `UTTERANCE`

### Relations
There are **19 unique relations**, including:
- `movie.initial_release_date`
- `movie.starring.actor`
- `movie.directed_by`
- `movie.genre`
- `actor.gender`
- `person.date_of_birth`
- `none`

These relations define the output space for the model.

---

## Task Formulation

This is a **multi-label classification** problem:
- An utterance can invoke **zero, one, or multiple relations**
- Example: asking for a *female actor* invokes both a starring relation and a gender relation

Because of this:
- Softmax-based losses are inappropriate
- Sigmoid-based losses are required

---

## Model Architecture

The model is a **Multilayer Perceptron (MLP)** with the following structure:

- Input layer (vectorized text)
- Hidden layer: 512 units
- Hidden layer: 256 units
- Dropout: 0.3
- Output layer: one neuron per relation
- Activation: ReLU or Leaky ReLU (varied by experiment)

This architecture was chosen for:
- Simplicity
- Interpretability
- Controlled experimentation

---

## Feature Representation

Text is vectorized using **CountVectorizer**, with experiments across:

### Word-level tokenization
- Captures semantic meaning
- Large vocabulary
- More sparse features

### Character-level tokenization (`char_wb`)
- Captures subword patterns
- Smaller, more robust vocabulary
- Particularly effective for short utterances

Stopwords were removed and vocabulary size was capped to reduce noise.

---

## Training Setup

- Optimizers tested:
  - Adam
  - AdamW
  - NAdam
  - Adagrad
- Loss functions:
  - **BCEWithLogitsLoss** (primary)
  - CrossEntropyLoss (tested, but unsuitable)
- Learning rates:
  - 0.01
  - 0.001
- Epochs:
  - 100
  - 1000

Train/test split:
- 80% training
- 20% testing

---

## Experiments and Findings

### Most Important Observations

1. **Loss function choice matters**
   - BCEWithLogitsLoss consistently outperformed CrossEntropy
   - This aligns with the multi-label nature of the task

2. **Learning rate dominates optimizer choice**
   - High learning rates caused oscillation
   - Low learning rates stabilized both accuracy and loss

3. **Optimizer choice had minimal impact**
   - Adam, AdamW, and NAdam produced similar results when learning rate was controlled

4. **Character-level tokenization was the biggest win**
   - `char_wb` analyzer produced the highest accuracy
   - Particularly effective for short, structured queries

5. **More epochs ≠ better**
   - Increasing to 1000 epochs often led to oscillation or mild overfitting
   - 100 epochs was generally sufficient

### Best Configuration
- Optimizer: AdamW
- Learning rate: 0.001
- Loss: BCEWithLogitsLoss
- Activation: Leaky ReLU
- Tokenization: char-level (`char_wb`)
- Epochs: 100

This achieved **~73.4% accuracy**, a large improvement over the baseline (~58%).

---

## Key Lessons Learned

- Model correctness depends on **problem formulation** (multi-label vs multi-class)
- Loss functions must match output semantics
- Feature representation can matter more than architecture
- Character-level models are often underappreciated for short-text NLP
- Stability (learning rate, loss) matters more than optimizer novelty

---

## Limitations

- No contextual embeddings (e.g., BERT)
- Simple MLP architecture
- No explicit handling of label correlations
- Accuracy used as primary metric (could be expanded to precision/recall per relation)

---

## Possible Improvements

If revisiting this project:
1. Use pretrained embeddings or transformers
2. Add per-relation precision/recall analysis
3. Model label dependencies explicitly
4. Apply threshold tuning per relation
5. Introduce cross-validation for more stable estimates

---

## References

- Dive into Deep Learning (Zhang et al.)
- PyTorch BCEWithLogitsLoss documentation
- Freebase schema
- Character vs Word Tokenization literature

---

## Notes to Future Me

This project is about **learning dynamics**, not squeezing out maximum accuracy.  
The most valuable insight was how *representation + loss + learning rate* dominated outcomes far more than architecture tweaks.
