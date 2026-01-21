# Relation Extraction / Utterance Classification (PyTorch MLP)

This project implements a supervised NLP model to classify natural-language **utterances** into one or more **semantic relations** from a movie-focused schema. The task is framed as **multi-label classification**, since a single utterance may invoke multiple relations.

This project was built as a learning exercise to understand text vectorization, multi-label learning, neural baselines, and systematic experimentation.

---

## Table of Contents
- Project Goal
- Dataset
- Task Framing
- Model
- Text Representation
- Training Setup
- Experiments
- Results Summary
- Key Takeaways
- Common Pitfalls / Notes to Future Me
- How to Reproduce
- Next Improvements

---

## Project Goal
Given a user utterance (e.g., a question about a movie), predict the correct relation(s) it refers to.  
Examples include relations like:
- movie release date
- movie director
- movie genre
- actors or characters in a movie

The core objective is to map free-form text to structured semantic relations.

---

## Dataset
The dataset consists of two CSV files:

- **Training data**: utterance IDs, utterances, and labeled relations
- **Test data**: utterance IDs and utterances only

There are **19 unique relations** in total, including a `none` label for utterances that do not map to any defined relation.

The dataset focuses on the **film domain**, using relations inspired by Freebase-style schemas.

---

## Task Framing

### Multi-label Classification
Some utterances can reference **multiple relations** at once. For example, a question may ask about both an actor and a movie attribute.

Because of this, the problem is framed as **multi-label classification**, not simple multi-class classification.

Implications:
- Each utterance can have zero, one, or multiple labels
- The model outputs a score per relation
- Labels are predicted independently using sigmoid-style outputs

---

## Model
The core model is a **Multilayer Perceptron (MLP)** implemented in PyTorch.

Architecture:
- Input layer: vectorized text features
- Hidden layer 1: 512 units
- Hidden layer 2: 256 units
- Dropout: 0.3
- Activation: ReLU or Leaky ReLU (experimented)

Why an MLP:
- Works well with sparse bag-of-words style features
- Can model non-linear decision boundaries
- Simple enough to reason about and debug

---

## Text Representation
Text is vectorized using **CountVectorizer**.

Key configuration options explored:
- Word-level tokenization
- Character-level tokenization using `char_wb`
- Stopword removal
- Vocabulary size limits

### Character-level tokenization
The largest performance gains came from switching to **character-level features**:
- More robust to spelling variation
- Smaller and denser feature space
- Better generalization on smaller datasets

---

## Training Setup

### Data Split
The training data is split into:
- 80% training
- 20% validation/testing

### Loss Function
The primary loss function is **BCEWithLogitsLoss**, which is appropriate for multi-label classification.

CrossEntropyLoss was also tested but performed poorly, as it assumes mutually exclusive classes.

### Optimizers
Optimizers explored:
- Adam
- AdamW
- NAdam
- Adagrad

AdamW generally produced the most stable results.

### Hyperparameters
Experiments varied:
- Learning rate (0.01 vs 0.001)
- Number of epochs (100 vs 1000)
- Activation function (ReLU vs Leaky ReLU)
- Tokenization strategy (word vs character)

---

## Experiments
Experiments were run by changing one component at a time to isolate its effect.

Key observations:
- Lower learning rates significantly reduced training instability
- Increasing epochs beyond a certain point led to oscillation and potential overfitting
- Character-level tokenization provided the single largest accuracy improvement
- Loss function choice mattered greatly due to the multi-label setup

---

## Results Summary
The best-performing configuration achieved approximately **73% prediction accuracy** on the validation set.

Best configuration highlights:
- AdamW optimizer
- Learning rate: 0.001
- 100 training epochs
- BCEWithLogitsLoss
- Leaky ReLU activation
- Character-level tokenization

---

## Key Takeaways
1. **Problem framing matters**: multi-label vs multi-class changes everything.
2. **Loss function choice is critical** for multi-label classification.
3. **Character-level features** can outperform word-level features on small datasets.
4. More training epochs does not always improve performance.
5. Simple models with good features can be very competitive.

---

## Common Pitfalls / Notes to Future Me
- Always confirm whether the task is multi-label or multi-class before choosing a loss.
- If training is unstable, check learning rate first.
- Tokenization strategy can dominate performance.
- Accuracy alone is insufficient; per-label metrics matter.
- Thresholding logits into labels is a non-trivial decision.

---

## How to Reproduce
1. Place the training and test CSV files in the project directory.
2. Vectorize utterances using CountVectorizer (try both word and character analyzers).
3. Train the MLP using BCEWithLogitsLoss.
4. Evaluate on the held-out validation set.
5. Generate predictions for the test set if required.

---

## Next Improvements
If revisiting this project:
- Add per-label precision, recall, and F1 metrics
- Tune decision thresholds per relation
- Try TF-IDF instead of raw counts
- Compare against linear baselines (Logistic Regression, linear SVM)
- Add cross-validation for more reliable comparisons
- Explore regularization (dropout rates, weight decay)

---

## Final Note
This project is intentionally simple in architecture but rich in learning value. The emphasis is on correct problem framing, disciplined experimentation, and understanding why design choices matter.
