# 🤖 AI Problem Solving Assignment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-orange?style=for-the-badge)
![Algorithm](https://img.shields.io/badge/AI-CSP%20%7C%20ML-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Department of Computer Science & Engineering**
**Artificial Intelligence — Problem Solving Assignment**

</div>

---

## 📁 Repository Structure

```
AI_ProblemSolving_<RegisterNumber>/
│
├── Problem_06_Sudoku_CSP/
│   ├── sudoku_solver.py        ← Main application (single-file)
│   ├── requirements.txt
│   └── screenshots/
│       ├── gameplay.png
│       └── ai_solve.png
│
├── Problem_19_Loan_Prediction/
│   ├── loan_prediction.py      ← Main application (single-file)
│   ├── requirements.txt
│   └── screenshots/
│       ├── approved.png
│       └── rejected.png
│
└── README.md
```

---

## 🧩 Problem 6 — Sudoku Solver using CSP

### 📌 Problem Description

Sudoku is a logic-based number placement puzzle played on a **9×9 grid**. Some cells are pre-filled, and the objective is to fill the remaining cells so that every row, every column, and every 3×3 sub-grid contains all of the digits from 1 to 9 **without repetition**.

This program provides:
- An **interactive GUI** where users solve the puzzle manually
- An **AI solver** that demonstrates the CSP algorithm step-by-step
- **Real-time validation** (Check / Hint / Reset)
- A **live timer** and CSP statistics (nodes explored, solve time)

### 🧠 Algorithm Used

#### Constraint Satisfaction Problem (CSP) — Backtracking + MRV Heuristic

Sudoku is naturally modelled as a **CSP**:

| CSP Component | Sudoku Mapping |
|---|---|
| **Variables** | Each of the 81 cells |
| **Domain** | Digits {1, 2, 3, …, 9} |
| **Constraints** | No repeated digit in any row, column, or 3×3 box |

**Search Strategy:**

```
function BACKTRACK(board):
    cell ← SELECT_UNASSIGNED_VAR(board)   ← MRV heuristic
    if cell is None:
        return SOLVED ✓

    for num in POSSIBLE_VALUES(cell):
        nodes_explored += 1
        if is_valid(board, cell, num):
            board[cell] ← num
            result ← BACKTRACK(board)
            if result ≠ FAILURE:
                return result
            board[cell] ← 0              ← Backtrack

    return FAILURE
```

**MRV (Minimum Remaining Values) Heuristic:**
> Always pick the cell with the **fewest legal values** remaining in its domain. This dramatically reduces the search space by catching failures early.

**Three CSP Constraints enforced at every step:**
1. ✅ Each **row** contains digits 1–9 without repetition
2. ✅ Each **column** contains digits 1–9 without repetition
3. ✅ Each **3×3 box** contains digits 1–9 without repetition

### ⚡ Performance

| Metric | Value |
|---|---|
| Nodes explored (Easy puzzle) | ~50–150 |
| Solve time | < 50 ms |
| Algorithm complexity | O(9^m) worst case, m = empty cells |
| With MRV heuristic | Near-linear in practice |

### 🖥️ Execution Steps

**Prerequisites:**
```bash
# Python 3.8+ required
# Tkinter is included with standard Python installations
python --version
```

**Run the application:**
```bash
# Navigate to the problem folder
cd Problem_06_Sudoku_CSP

# Run directly — no pip install needed!
python sudoku_solver.py
```

**How to use:**
1. Launch the app — a pre-filled easy puzzle loads automatically
2. Click any **white cell** and type a digit (1–9) to fill it in
3. Press **CHECK** to validate your current entries (wrong cells highlight red)
4. Press **💡 HINT** to reveal one correct cell (use sparingly!)
5. Press **⚡ AI SOLVE** to watch the CSP algorithm solve it step-by-step
6. Press **↺ RESET** to clear your entries and try again
7. Use the **dropdown** to switch between 3 different puzzles

### 📊 Sample Output

```
Puzzle: Easy 1 (Classic)

Before solving:               After AI Solve (CSP):
5 3 _ | _ 7 _ | _ _ _         5 3 4 | 6 7 8 | 9 1 2
6 _ _ | 1 9 5 | _ _ _         6 7 2 | 1 9 5 | 3 4 8
_ 9 8 | _ _ _ | _ 6 _         1 9 8 | 3 4 2 | 5 6 7
------+-------+------         ------+-------+------
8 _ _ | _ 6 _ | _ _ 3         8 5 9 | 7 6 1 | 4 2 3
4 _ _ | 8 _ 3 | _ _ 1         4 2 6 | 8 5 3 | 7 9 1
7 _ _ | _ 2 _ | _ _ 6         7 1 3 | 9 2 4 | 8 5 6
------+-------+------         ------+-------+------
_ 6 _ | _ _ _ | 2 8 _         9 6 1 | 5 3 7 | 2 8 4
_ _ _ | 4 1 9 | _ _ 5         2 8 7 | 4 1 9 | 6 3 5
_ _ _ | _ 8 _ | _ 7 9         3 4 5 | 2 8 6 | 1 7 9

CSP Statistics:
  ✅ Solved: YES
  🔢 Nodes explored: 51
  ⏱ Solve time: 12.4 ms
  📐 Algorithm: Backtracking + MRV Heuristic
```

---

## 🏦 Problem 19 — Loan Approval Prediction System

### 📌 Problem Description

A bank wants to automate its loan approval process. Given an applicant's financial profile — income, credit score, loan amount requested, employment history, existing debt, and assets — the system **predicts whether the loan should be approved or rejected**.

This program provides:
- **Two ML classifiers** trained from scratch (no sklearn): Logistic Regression and Random Forest
- An **interactive GUI** with sliders for all 6 input features
- **Real-time prediction** with confidence probability bar
- **Model comparison** showing both classifiers' outputs side-by-side
- **Feature importance** chart showing which factors matter most

### 🧠 Algorithm Used

#### Model 1 — Logistic Regression (Gradient Descent)

Binary classifier: predicts P(approved) ∈ [0, 1].

```
Hypothesis:   h(x) = σ(w·x + b)      where σ(z) = 1 / (1 + e^(-z))

Loss:         L = -[y·log(h) + (1-y)·log(1-h)]   ← Binary Cross-Entropy

Update rule:  w_j ← w_j - α · (h(x) - y) · x_j
              b   ← b   - α · (h(x) - y)

Hyperparams:  α = 0.05 (learning rate), epochs = 500
```

#### Model 2 — Random Forest (Bootstrap Aggregation)

Ensemble of **25 Decision Trees**, each trained on a bootstrap sample.

```
for i in 1..25:
    bootstrap_sample ← random sample WITH replacement from training set
    feature_subset   ← random √n_features from all features  ← reduces correlation
    tree[i]          ← train DecisionTree(bootstrap_sample, feature_subset)
                       using Gini Impurity to find best splits

predict(x):
    votes ← [tree[i].predict_proba(x) for i in 1..25]
    return average(votes)                                     ← soft voting
```

**Decision Tree split criterion — Gini Impurity:**
```
Gini(S) = 1 - Σ p_i²

Information Gain = Gini(parent) - Σ (|child|/|parent|) · Gini(child)
```

#### Ensemble Decision
```
Final Probability = (LR_probability + RF_probability) / 2

Approved  if  Final_Probability ≥ 0.50
Rejected  if  Final_Probability <  0.50
```

### 📐 Features Used

| # | Feature | Range | Impact |
|---|---|---|---|
| 1 | Annual Income | ₹2L – ₹50L | High ↑ |
| 2 | Credit Score | 300 – 900 | Highest ↑↑ |
| 3 | Loan Amount | ₹1L – ₹40L | Medium ↓ |
| 4 | Employment Duration | 0 – 30 yrs | Medium ↑ |
| 5 | Existing Debt Ratio | 5% – 80% | High ↓ |
| 6 | Total Assets | ₹0 – ₹100L | Medium ↑ |

### 🖥️ Execution Steps

**Prerequisites:**
```bash
# Python 3.8+ required
# Uses ONLY Python standard library — no pip install needed!
python --version
```

**Run the application:**
```bash
# Navigate to the problem folder
cd Problem_19_Loan_Prediction

# Run directly!
python loan_prediction.py
```

**How to use:**
1. Launch the app — models train automatically on startup (~2–5 seconds)
2. Wait for **"✅ Training complete!"** in the status bar
3. Use the **sliders** to set the applicant's financial details
4. Click **⚡ PREDICT LOAN DECISION**
5. View the result: APPROVED ✅ or REJECTED ❌ with confidence %
6. The **probability bar** shows approval likelihood (green = approved, red = rejected)
7. The **model comparison** section shows what each classifier individually predicted
8. The **feature importance** chart shows which factors most influenced the Random Forest

### 📊 Sample Output

```
============================================================
         LOAN APPROVAL PREDICTION SYSTEM
         ML Classification | LR + Random Forest
============================================================

Dataset: 1,000 synthetic samples | Train: 800 | Test: 200

📈 Model Performance:
  Logistic Regression Accuracy : 84.5%
  Random Forest Accuracy       : 87.0%

  Confusion Matrix (Random Forest on Test Set):
  ┌─────────────┬──────────┬──────────┐
  │             │ Pred: NO │ Pred: YES│
  ├─────────────┼──────────┼──────────┤
  │ Actual: NO  │  TN: 89  │  FP: 11  │
  │ Actual: YES │  FN: 15  │  TP: 85  │
  └─────────────┴──────────┴──────────┘

------------------------------------------------------------
APPLICANT PROFILE:
  Income            : ₹35.0 Lakhs/yr
  Credit Score      : 750
  Loan Amount       : ₹12.0 Lakhs
  Employment        : 8.0 years
  Existing Debt     : 20%
  Total Assets      : ₹45.0 Lakhs

PREDICTION RESULTS:
  Logistic Regression → APPROVED  (87.3% confidence)
  Random Forest       → APPROVED  (91.2% confidence)
  Ensemble Average    → APPROVED ✅ (89.3% confidence)

KEY FACTORS:
  ✅ Credit score 750 — excellent creditworthiness
  ✅ Debt ratio 20% — well within safe limits
  ✅ Loan-to-income ratio 0.34 — manageable
============================================================
```

---

## 🛠️ Requirements

Both projects use **Python standard library only** — no external packages required.

```
# requirements.txt (same for both problems)
# Python 3.8+ with standard library

tkinter    ← GUI (bundled with Python)
math       ← sigmoid, sqrt (built-in)
random     ← dataset generation, bootstrap (built-in)
copy       ← deep copy board states (built-in)
time       ← timer, performance measurement (built-in)
threading  ← non-blocking AI solve animation (built-in)
```

> ⚠️ On some Linux distros, Tkinter may need to be installed separately:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## 👥 Team Details

| Field | Details |
|---|---|
| **Team Size** | 2 Members |
| **Subject** | Artificial Intelligence |
| **Assignment Type** | Problem Solving (GitHub Submission) |
| **Submission Deadline** | 25th April 2026 |
| **Language** | Python 3.8+ |

---

## 📜 License

This project is submitted as part of an academic assignment. MIT License — free to use for educational purposes.

---

<div align="center">

Made with 🧠 using AI Problem Solving Techniques

*CSP · Backtracking · MRV · Logistic Regression · Random Forest · Gini Impurity · Bootstrap Aggregation*

</div>
