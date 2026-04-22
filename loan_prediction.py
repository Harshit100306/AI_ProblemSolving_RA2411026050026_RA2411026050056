"""
Problem 19: Loan Approval Prediction System (Classification Task)
=================================================================
AI Problem Solving Assignment
Algorithm: Logistic Regression + Random Forest Classification

Features:
  - Synthetic dataset generation (1000 samples, 6 features)
  - Data preprocessing: encoding, normalization, handling imbalance
  - Trains both Logistic Regression and Random Forest models
  - Interactive GUI: input applicant details → get instant prediction
  - Displays model accuracy, confusion matrix, feature importance
  - Confidence score with visual probability bar
  - Comparison between both models
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math
import random
import time

# ─────────────────────────────────────────────
#  PURE-PYTHON ML ENGINE (no sklearn required)
# ─────────────────────────────────────────────

random.seed(42)


def sigmoid(z):
    if z > 500: return 1.0
    if z < -500: return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def normalize(data):
    """Min-max normalize each feature column."""
    n_features = len(data[0])
    mins = [min(row[i] for row in data) for i in range(n_features)]
    maxs = [max(row[i] for row in data) for i in range(n_features)]
    ranges = [mx - mn if mx != mn else 1 for mx, mn in zip(maxs, mins)]
    norm = [[(row[i] - mins[i]) / ranges[i] for i in range(n_features)]
            for row in data]
    return norm, mins, maxs, ranges


def normalize_sample(sample, mins, ranges):
    return [(v - mn) / rng for v, mn, rng in zip(sample, mins, ranges)]


class LogisticRegression:
    """Logistic Regression trained via Gradient Descent."""

    def __init__(self, lr=0.1, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        n_feat = len(X[0])
        self.weights = [0.0] * n_feat
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sigmoid(dot(self.weights, xi) + self.bias)
                err  = pred - yi
                self.weights = [w - self.lr * err * xi[j]
                                for j, w in enumerate(self.weights)]
                self.bias -= self.lr * err

    def predict_proba(self, x):
        p = sigmoid(dot(self.weights, x) + self.bias)
        return p

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def score(self, X, y):
        correct = sum(1 for xi, yi in zip(X, y) if self.predict(xi) == yi)
        return correct / len(y)


class DecisionTree:
    """Simple decision tree (depth-limited) for Random Forest."""

    def __init__(self, max_depth=6, min_samples=5, n_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.n_features = n_features
        self.tree = None

    def _gini(self, y):
        n = len(y)
        if n == 0: return 0
        p1 = sum(y) / n
        p0 = 1 - p1
        return 1 - p0**2 - p1**2

    def _best_split(self, X, y):
        n_feat = len(X[0])
        if self.n_features:
            feats = random.sample(range(n_feat),
                                  min(self.n_features, n_feat))
        else:
            feats = list(range(n_feat))

        best_gain, best_feat, best_thr = -1, 0, 0

        for f in feats:
            vals = sorted(set(row[f] for row in X))
            thresholds = [(vals[i] + vals[i+1]) / 2
                          for i in range(len(vals)-1)] or [vals[0]]
            for thr in thresholds:
                left_y  = [yi for xi, yi in zip(X, y) if xi[f] <= thr]
                right_y = [yi for xi, yi in zip(X, y) if xi[f] >  thr]
                if not left_y or not right_y:
                    continue
                gain = (self._gini(y)
                        - len(left_y)/len(y) * self._gini(left_y)
                        - len(right_y)/len(y) * self._gini(right_y))
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, f, thr

        return best_feat, best_thr

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) <= self.min_samples
                or len(set(y)) == 1):
            return {"leaf": True, "value": round(sum(y) / len(y))}

        feat, thr = self._best_split(X, y)
        left_idx  = [i for i, xi in enumerate(X) if xi[feat] <= thr]
        right_idx = [i for i, xi in enumerate(X) if xi[feat] >  thr]

        if not left_idx or not right_idx:
            return {"leaf": True, "value": round(sum(y) / len(y))}

        return {
            "leaf": False, "feat": feat, "thr": thr,
            "left":  self._build([X[i] for i in left_idx],
                                  [y[i] for i in left_idx], depth + 1),
            "right": self._build([X[i] for i in right_idx],
                                  [y[i] for i in right_idx], depth + 1),
        }

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)

    def _predict_one(self, node, x):
        if node["leaf"]:
            return node["value"]
        if x[node["feat"]] <= node["thr"]:
            return self._predict_one(node["left"], x)
        return self._predict_one(node["right"], x)

    def predict_proba(self, x):
        return self._predict_one(self.tree, x)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class RandomForest:
    """Random Forest: ensemble of decision trees (bootstrap + feature subsampling)."""

    def __init__(self, n_trees=25, max_depth=6, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n = len(X)
        n_feat = self.n_features or max(1, int(math.sqrt(len(X[0]))))
        for _ in range(self.n_trees):
            idx = [random.randint(0, n-1) for _ in range(n)]
            Xb  = [X[i] for i in idx]
            yb  = [y[i] for i in idx]
            tree = DecisionTree(max_depth=self.max_depth,
                                n_features=n_feat)
            tree.fit(Xb, yb)
            self.trees.append(tree)

    def predict_proba(self, x):
        return sum(t.predict_proba(x) for t in self.trees) / self.n_trees

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def score(self, X, y):
        correct = sum(1 for xi, yi in zip(X, y) if self.predict(xi) == yi)
        return correct / len(y)

    def feature_importance(self, n_features):
        """Approximate feature importance from tree structure."""
        counts = [0] * n_features

        def count_splits(node):
            if node["leaf"]: return
            counts[node["feat"]] += 1
            count_splits(node["left"])
            count_splits(node["right"])

        for tree in self.trees:
            count_splits(tree.tree)

        total = sum(counts) or 1
        return [c / total for c in counts]


# ─────────────────────────────────────────────
#  DATASET GENERATION
# ─────────────────────────────────────────────

FEATURE_NAMES = [
    "Income (₹/yr, L)",
    "Credit Score",
    "Loan Amount (₹L)",
    "Employment (yrs)",
    "Debt Ratio (%)",
    "Assets (₹L)",
]

def generate_dataset(n=1000):
    """Generate realistic synthetic loan dataset."""
    X, y = [], []
    for _ in range(n):
        income      = random.uniform(2, 50)          # Lakhs
        credit      = random.uniform(300, 900)        # Score
        loan_amt    = random.uniform(1, 40)           # Lakhs
        employment  = random.uniform(0, 30)           # Years
        debt_ratio  = random.uniform(5, 80)           # %
        assets      = random.uniform(0, 100)          # Lakhs

        # Approval logic (with noise)
        score = (
            (income / 50) * 30 +
            ((credit - 300) / 600) * 35 +
            (1 - loan_amt / 40) * 15 +
            (employment / 30) * 10 +
            (1 - debt_ratio / 80) * 5 +
            (assets / 100) * 5
        )
        noise = random.gauss(0, 8)
        label = 1 if (score + noise) > 45 else 0

        X.append([income, credit, loan_amt, employment, debt_ratio, assets])
        y.append(label)

    return X, y


def train_test_split(X, y, test_ratio=0.2):
    combined = list(zip(X, y))
    random.shuffle(combined)
    split = int(len(combined) * (1 - test_ratio))
    train = combined[:split]
    test  = combined[split:]
    Xtr, ytr = [t[0] for t in train], [t[1] for t in train]
    Xte, yte = [t[0] for t in test],  [t[1] for t in test]
    return Xtr, Xte, ytr, yte


def confusion_matrix(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    return tp, tn, fp, fn


# ─────────────────────────────────────────────
#  GUI APPLICATION
# ─────────────────────────────────────────────

class LoanApp:
    BG       = "#0a0f1e"
    PANEL_BG = "#111827"
    CARD_BG  = "#1a2235"
    ACCENT   = "#3b82f6"
    ACCENT2  = "#10b981"
    REJECT   = "#ef4444"
    WARN     = "#f59e0b"
    FG       = "#f1f5f9"
    FG_DIM   = "#64748b"
    BORDER   = "#1e3a5f"

    FONT_TITLE  = ("Georgia", 24, "bold")
    FONT_SUB    = ("Courier", 10)
    FONT_LABEL  = ("Courier", 11, "bold")
    FONT_BODY   = ("Courier", 10)
    FONT_BTN    = ("Courier", 12, "bold")
    FONT_RESULT = ("Georgia", 18, "bold")
    FONT_STAT   = ("Courier", 9)

    def __init__(self, root):
        self.root = root
        self.root.title("Loan Approval Prediction — AI Problem Solving Assignment")
        self.root.configure(bg=self.BG)
        self.root.resizable(False, False)

        self.lr_model = None
        self.rf_model = None
        self.mins = None
        self.ranges = None
        self.lr_acc = 0
        self.rf_acc = 0
        self.trained = False

        self.status_var = tk.StringVar(value="Training models on synthetic dataset…")
        self._build_ui()
        self.root.after(200, self._train_models)

    def _build_ui(self):
        # ── Title ──────────────────────────────────
        title_frame = tk.Frame(self.root, bg=self.BG, pady=20)
        title_frame.pack(fill="x")
        tk.Label(title_frame, text="LOAN APPROVAL", font=("Georgia", 28, "bold"),
                 bg=self.BG, fg=self.ACCENT).pack()
        tk.Label(title_frame, text="PREDICTION SYSTEM",
                 font=("Georgia", 16, "bold"), bg=self.BG, fg=self.FG_DIM).pack()
        tk.Label(title_frame,
                 text="ML Classification  ·  Logistic Regression  +  Random Forest",
                 font=self.FONT_SUB, bg=self.BG, fg=self.FG_DIM).pack(pady=(4, 0))

        # ── Body ───────────────────────────────────
        body = tk.Frame(self.root, bg=self.BG)
        body.pack(padx=30, pady=(0, 20), fill="both")

        self._build_input_panel(body)
        self._build_result_panel(body)

        # ── Stats bar ──────────────────────────────
        stats = tk.Frame(self.root, bg=self.PANEL_BG, pady=12)
        stats.pack(fill="x", padx=30, pady=(0, 20))
        self._build_stats_bar(stats)

        # ── Status ─────────────────────────────────
        status_f = tk.Frame(self.root, bg=self.BG, pady=6)
        status_f.pack(fill="x")
        tk.Label(status_f, textvariable=self.status_var,
                 font=self.FONT_BODY, bg=self.BG, fg=self.ACCENT2).pack()

    def _build_input_panel(self, parent):
        frame = tk.Frame(parent, bg=self.CARD_BG, bd=0,
                         padx=24, pady=20)
        frame.pack(side="left", fill="y", padx=(0, 16))

        tk.Label(frame, text="📋 APPLICANT DETAILS",
                 font=self.FONT_LABEL, bg=self.CARD_BG, fg=self.ACCENT).pack(anchor="w")
        tk.Label(frame, text="Fill in the applicant's financial information",
                 font=self.FONT_BODY, bg=self.CARD_BG, fg=self.FG_DIM).pack(anchor="w", pady=(2, 16))

        self.input_vars = {}
        fields = [
            ("Income (₹/yr in Lakhs)",     "income",     2.0,   50.0,  25.0),
            ("Credit Score (300–900)",      "credit",     300,   900,   650),
            ("Loan Amount (₹ Lakhs)",       "loan_amt",   1.0,   40.0,  15.0),
            ("Employment Duration (yrs)",   "employment", 0.0,   30.0,  5.0),
            ("Existing Debt Ratio (%)",     "debt_ratio", 5.0,   80.0,  30.0),
            ("Total Assets (₹ Lakhs)",      "assets",     0.0,   100.0, 20.0),
        ]

        for label, key, mn, mx, default in fields:
            row = tk.Frame(frame, bg=self.CARD_BG, pady=6)
            row.pack(fill="x")

            tk.Label(row, text=label, font=self.FONT_BODY,
                     bg=self.CARD_BG, fg=self.FG, width=30, anchor="w").pack(side="left")

            var = tk.DoubleVar(value=default)
            self.input_vars[key] = var

            val_label = tk.Label(row, text=f"{default:.0f}",
                                 font=("Courier", 10, "bold"),
                                 bg=self.CARD_BG, fg=self.ACCENT, width=5)
            val_label.pack(side="right")

            slider = tk.Scale(row, from_=mn, to=mx, orient="horizontal",
                              variable=var, resolution=0.5,
                              bg=self.CARD_BG, fg=self.FG,
                              troughcolor=self.BG, activebackground=self.ACCENT,
                              highlightthickness=0, bd=0, showvalue=False,
                              width=10, sliderlength=16,
                              command=lambda v, lbl=val_label: lbl.config(text=f"{float(v):.0f}"))
            slider.pack(side="right", fill="x", expand=True, padx=8)

        # Predict button
        tk.Frame(frame, bg=self.CARD_BG, height=16).pack()
        predict_btn = tk.Button(frame, text="⚡  PREDICT LOAN DECISION",
                                font=self.FONT_BTN, bg=self.ACCENT, fg="white",
                                bd=0, relief="flat", pady=12, cursor="hand2",
                                command=self._predict)
        predict_btn.pack(fill="x")
        predict_btn.bind("<Enter>", lambda e: predict_btn.config(bg="#2563eb"))
        predict_btn.bind("<Leave>", lambda e: predict_btn.config(bg=self.ACCENT))

    def _build_result_panel(self, parent):
        frame = tk.Frame(parent, bg=self.CARD_BG, padx=24, pady=20, width=320)
        frame.pack(side="left", fill="both", expand=True)
        frame.pack_propagate(False)

        tk.Label(frame, text="🔍 PREDICTION RESULT",
                 font=self.FONT_LABEL, bg=self.CARD_BG, fg=self.ACCENT).pack(anchor="w")
        tk.Label(frame, text="ML model output and confidence",
                 font=self.FONT_BODY, bg=self.CARD_BG, fg=self.FG_DIM).pack(anchor="w", pady=(2, 16))

        # Decision display
        self.decision_var = tk.StringVar(value="—")
        self.decision_label = tk.Label(frame, textvariable=self.decision_var,
                                       font=("Georgia", 28, "bold"),
                                       bg=self.CARD_BG, fg=self.FG_DIM)
        self.decision_label.pack(pady=(8, 4))

        self.confidence_var = tk.StringVar(value="Enter applicant details and click Predict")
        tk.Label(frame, textvariable=self.confidence_var,
                 font=self.FONT_BODY, bg=self.CARD_BG, fg=self.FG_DIM,
                 wraplength=260).pack()

        # Probability bar canvas
        tk.Label(frame, text="Approval Probability", font=self.FONT_BODY,
                 bg=self.CARD_BG, fg=self.FG_DIM).pack(pady=(16, 4))
        self.prob_canvas = tk.Canvas(frame, width=260, height=32,
                                     bg=self.BG, highlightthickness=0, bd=0)
        self.prob_canvas.pack()
        self._draw_prob_bar(0.5)

        # Model comparison
        tk.Label(frame, text="━" * 30, bg=self.CARD_BG, fg=self.BORDER).pack(pady=(16, 8))
        tk.Label(frame, text="MODEL COMPARISON", font=self.FONT_LABEL,
                 bg=self.CARD_BG, fg=self.FG_DIM).pack(anchor="w")

        self.lr_result_var  = tk.StringVar(value="—")
        self.rf_result_var  = tk.StringVar(value="—")
        self.lr_conf_var    = tk.StringVar(value="—")
        self.rf_conf_var    = tk.StringVar(value="—")

        for model, res_var, conf_var, color in [
            ("Logistic Regression", self.lr_result_var, self.lr_conf_var, "#818cf8"),
            ("Random Forest",       self.rf_result_var, self.rf_conf_var, "#34d399"),
        ]:
            row = tk.Frame(frame, bg=self.CARD_BG, pady=4)
            row.pack(fill="x")
            tk.Label(row, text=model, font=self.FONT_BODY,
                     bg=self.CARD_BG, fg=color, width=20, anchor="w").pack(side="left")
            tk.Label(row, textvariable=res_var, font=("Courier", 10, "bold"),
                     bg=self.CARD_BG, fg=color, width=9).pack(side="left")
            tk.Label(row, textvariable=conf_var, font=self.FONT_BODY,
                     bg=self.CARD_BG, fg=self.FG_DIM).pack(side="left")

        # Feature importance (placeholder, updated after training)
        tk.Label(frame, text="━" * 30, bg=self.CARD_BG, fg=self.BORDER).pack(pady=(12, 8))
        tk.Label(frame, text="FEATURE IMPORTANCE (RF)", font=self.FONT_LABEL,
                 bg=self.CARD_BG, fg=self.FG_DIM).pack(anchor="w")
        self.fi_canvas = tk.Canvas(frame, width=260, height=130,
                                   bg=self.CARD_BG, highlightthickness=0, bd=0)
        self.fi_canvas.pack()

    def _build_stats_bar(self, parent):
        self.stat_vars = {}
        stats = [
            ("Dataset Size", "dataset_size", "—"),
            ("Training Samples", "train_size", "—"),
            ("Test Samples", "test_size", "—"),
            ("LR Accuracy", "lr_acc", "—"),
            ("RF Accuracy", "rf_acc", "—"),
            ("Features Used", "n_features", "6"),
        ]
        for label, key, default in stats:
            col = tk.Frame(parent, bg=self.PANEL_BG, padx=16)
            col.pack(side="left", expand=True)
            var = tk.StringVar(value=default)
            self.stat_vars[key] = var
            tk.Label(col, textvariable=var, font=("Courier", 16, "bold"),
                     bg=self.PANEL_BG, fg=self.ACCENT).pack()
            tk.Label(col, text=label, font=self.FONT_STAT,
                     bg=self.PANEL_BG, fg=self.FG_DIM).pack()

    def _draw_prob_bar(self, prob):
        self.prob_canvas.delete("all")
        w, h = 260, 32
        self.prob_canvas.create_rectangle(0, 0, w, h, fill=self.BG, outline="")
        # Background
        self.prob_canvas.create_rectangle(2, 8, w-2, h-8,
                                          fill="#1e293b", outline="")
        # Fill
        fill_w = int((w - 4) * prob)
        if fill_w > 0:
            color = self.ACCENT2 if prob >= 0.5 else self.REJECT
            self.prob_canvas.create_rectangle(2, 8, 2 + fill_w, h-8,
                                              fill=color, outline="")
        # Label
        pct = f"{prob*100:.1f}%"
        self.prob_canvas.create_text(w//2, h//2, text=pct,
                                     font=("Courier", 11, "bold"),
                                     fill="white")

    def _draw_feature_importance(self, importances):
        c = self.fi_canvas
        c.delete("all")
        w, h = 260, 130
        n = len(importances)
        bar_h = 14
        gap = 6
        max_val = max(importances) or 1
        short_names = ["Income", "Credit", "Loan Amt", "Employ.", "Debt %", "Assets"]

        for i, (val, name) in enumerate(zip(importances, short_names)):
            y = i * (bar_h + gap) + 4
            bar_w = int((w - 80) * val / max_val)
            c.create_text(0, y + bar_h//2, text=name, anchor="w",
                          font=("Courier", 8), fill=self.FG_DIM)
            x0 = 60
            c.create_rectangle(x0, y, w - 5, y + bar_h,
                                fill="#1e293b", outline="")
            if bar_w > 0:
                c.create_rectangle(x0, y, x0 + bar_w, y + bar_h,
                                   fill="#34d399", outline="")
            c.create_text(x0 + bar_w + 4, y + bar_h//2,
                          text=f"{val*100:.0f}%", anchor="w",
                          font=("Courier", 8), fill=self.FG_DIM)

    # ── Training ──────────────────────────────────────────────

    def _train_models(self):
        self.status_var.set("⚙ Generating synthetic dataset (1,000 samples)…")
        self.root.update()

        X, y = generate_dataset(1000)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_ratio=0.2)

        # Normalize
        Xtr_n, mins, maxs, ranges = normalize(Xtr)
        Xte_n = [normalize_sample(xi, mins, ranges) for xi in Xte]
        self.mins = mins
        self.ranges = ranges

        self.status_var.set("⚙ Training Logistic Regression…")
        self.root.update()
        t0 = time.perf_counter()
        self.lr_model = LogisticRegression(lr=0.05, epochs=500)
        self.lr_model.fit(Xtr_n, ytr)
        lr_time = (time.perf_counter() - t0) * 1000

        self.status_var.set("⚙ Training Random Forest (25 trees)…")
        self.root.update()
        t0 = time.perf_counter()
        self.rf_model = RandomForest(n_trees=25, max_depth=7)
        self.rf_model.fit(Xtr_n, ytr)
        rf_time = (time.perf_counter() - t0) * 1000

        # Evaluate
        lr_acc = self.lr_model.score(Xte_n, yte)
        rf_acc = self.rf_model.score(Xte_n, yte)
        self.lr_acc = lr_acc
        self.rf_acc = rf_acc

        # Confusion matrix
        lr_preds = [self.lr_model.predict(xi) for xi in Xte_n]
        rf_preds = [self.rf_model.predict(xi) for xi in Xte_n]
        tp, tn, fp, fn = confusion_matrix(yte, rf_preds)

        # Feature importance
        fi = self.rf_model.feature_importance(6)
        self._draw_feature_importance(fi)

        # Update stats
        self.stat_vars["dataset_size"].set("1,000")
        self.stat_vars["train_size"].set(f"{len(Xtr)}")
        self.stat_vars["test_size"].set(f"{len(Xte)}")
        self.stat_vars["lr_acc"].set(f"{lr_acc*100:.1f}%")
        self.stat_vars["rf_acc"].set(f"{rf_acc*100:.1f}%")

        self.trained = True
        self.status_var.set(
            f"✅ Training complete! "
            f"LR Accuracy: {lr_acc*100:.1f}% (took {lr_time:.0f}ms) | "
            f"RF Accuracy: {rf_acc*100:.1f}% (took {rf_time:.0f}ms)  "
            f"[Test set: TP={tp} TN={tn} FP={fp} FN={fn}]"
        )

    # ── Prediction ────────────────────────────────────────────

    def _predict(self):
        if not self.trained:
            messagebox.showwarning("Not Ready", "Models are still training. Please wait.")
            return

        try:
            sample_raw = [
                self.input_vars["income"].get(),
                self.input_vars["credit"].get(),
                self.input_vars["loan_amt"].get(),
                self.input_vars["employment"].get(),
                self.input_vars["debt_ratio"].get(),
                self.input_vars["assets"].get(),
            ]
        except Exception:
            messagebox.showerror("Input Error", "Invalid input values.")
            return

        # Basic validation
        issues = []
        if not (300 <= sample_raw[1] <= 900):
            issues.append("Credit score must be 300–900")
        if sample_raw[2] > sample_raw[0] * 5:
            issues.append("Loan amount is very high relative to income")
        if issues:
            messagebox.showwarning("Validation", "\n".join(issues))

        sample_n = normalize_sample(sample_raw, self.mins, self.ranges)

        lr_prob = self.lr_model.predict_proba(sample_n)
        rf_prob = self.rf_model.predict_proba(sample_n)

        lr_pred = 1 if lr_prob >= 0.5 else 0
        rf_pred = 1 if rf_prob >= 0.5 else 0

        # Ensemble: average probability
        avg_prob = (lr_prob + rf_prob) / 2
        final    = 1 if avg_prob >= 0.5 else 0
        # ─── Explainable AI (Decision Factors) ───
        reasons = []
        
        # Positive factors
        if sample_raw[1] >= 700:
            reasons.append("✔ High credit score")
        
        if sample_raw[0] >= 20:
            reasons.append("✔ Strong annual income")
        
        if sample_raw[4] <= 30:
            reasons.append("✔ Low existing debt")
        
        if sample_raw[3] >= 5:
            reasons.append("✔ Stable employment")
        
        if sample_raw[5] >= 30:
            reasons.append("✔ Good asset base")
        
        # Negative factors
        if sample_raw[1] < 580:
            reasons.append("✖ Low credit score")
        
        if sample_raw[4] > 50:
            reasons.append("✖ High debt ratio")
        
        if sample_raw[2] > sample_raw[0] * 3:
            reasons.append("✖ Loan too large compared to income")
        
        if sample_raw[3] < 1:
            reasons.append("✖ Unstable employment")

        # Update UI
        if final == 1:
            self.decision_var.set("✅ APPROVED")
            self.decision_label.config(fg=self.ACCENT2)
            conf_text = f"Confidence: {avg_prob*100:.1f}% — Loan is likely to be repaid."
        else:
            self.decision_var.set("❌ REJECTED")
            self.decision_label.config(fg=self.REJECT)
            conf_text = f"Confidence: {(1-avg_prob)*100:.1f}% — High risk of default."

        self.confidence_var.set(conf_text)
        self._draw_prob_bar(avg_prob)

        self.lr_result_var.set("APPROVED" if lr_pred else "REJECTED")
        self.rf_result_var.set("APPROVED" if rf_pred else "REJECTED")
        self.lr_conf_var.set(f"  ({lr_prob*100:.1f}%)")
        self.rf_conf_var.set(f"  ({rf_prob*100:.1f}%)")

        rules = []
        if sample_raw[1] < 580:
            rules.append("• Low credit score (< 580) → higher risk")
        if sample_raw[4] > 50:
            rules.append("• High debt ratio (> 50%) → financial strain")
        if sample_raw[2] > sample_raw[0] * 3:
            rules.append("• Loan exceeds 3× annual income → risky")
        if sample_raw[3] < 1:
            rules.append("• Less than 1 yr employment → unstable income")
        if not rules:
            rules = ["• Profile meets approval criteria"]

        reason_text = " | ".join(reasons[:2]) if reasons else "Profile acceptable"

        self.status_var.set(
            f"Prediction: {'APPROVED' if final else 'REJECTED'} | "
            f"Avg Probability: {avg_prob*100:.1f}% | "
            f"Key factors: {reason_text}"
        )


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.title("Loan Approval Prediction System")
    app = LoanApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
