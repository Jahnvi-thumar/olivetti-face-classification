#!/usr/bin/env python3
"""Reproduce Q2 boosting figures/tables from the assignment.

Run from repository root: python scripts/plot_q2.py
Requires olivetti_split.npz in the repository root and the same ROLL_NO seed
used to create it. Writes all figures to C:/Users/JAHANVI/ml_assignment_2/figure and results/ without requiring a GUI.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
SPLIT_PATH = ROOT / "olivetti_split.npz"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
DEPTHS = [1, 2, 3, 4]
ROUND_COUNTS = [10, 25, 50, 100]
B_VALUES = [25, 50, 100]
# Keep aligned with the assignment's Q1/Q3 frozen settings.
TREE_DEPTH, MIN_SAMPLES_LEAF, B_FINAL = None, 4, 50
SEED_FALLBACK = 26016


def load_data():
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SPLIT_PATH}. Run the Q1 split setup first and keep "
            "olivetti_split.npz in the repository root."
        )
    faces = fetch_olivetti_faces()
    X, y = faces.images.reshape(len(faces.images), -1), faces.target
    with np.load(SPLIT_PATH) as s:
        train_idx, val_idx, test_idx = s["train_indices"], s["val_indices"], s["test_indices"]
        seed = int(s["seed"]) if "seed" in s else SEED_FALLBACK
    if len(train_idx) != 240 or len(val_idx) != 80 or len(test_idx) != 80:
        raise ValueError("Saved split must contain 240 train, 80 validation and 80 test images.")
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx], seed


def samme(X_train, y_train, max_depth, n_rounds, seed):
    """SAMME using the notebook's multiclass update rule and rejection threshold."""
    classes = np.unique(y_train)
    K, n = len(classes), len(y_train)
    weights = np.ones(n, dtype=float) / n
    learners, alphas = [], []
    hist = {"weighted_error": [], "learner_weights": [], "max_sample_weight": [],
            "top10_weight": [], "sample_weights_by_round": [], "training_predictions": []}
    for t in range(n_rounds):
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=seed + t)
        tree.fit(X_train, y_train, sample_weight=weights)
        pred = tree.predict(X_train)
        hist["training_predictions"].append(pred.copy())
        wrong = pred != y_train
        error = float(weights[wrong].sum())
        # A weak learner must beat random multiclass prediction.
        if error >= 1 - 1 / K:
            print(f"SAMME depth={max_depth}: rejected round {t+1}, error={error:.5f}")
            break
        alpha = np.log((1 - max(error, 1e-12)) / max(error, 1e-12)) + np.log(K - 1)
        learners.append(tree)
        alphas.append(float(alpha))
        hist["weighted_error"].append(error)
        hist["learner_weights"].append(float(alpha))
        if error > 0:
            weights[wrong] *= np.exp(alpha)
            weights /= weights.sum()
        hist["sample_weights_by_round"].append(weights.copy())
        hist["max_sample_weight"].append(float(weights.max()))
        hist["top10_weight"].append(float(np.sort(weights)[-min(10, n):].sum()))
        if error == 0:
            break
    return {"learners": learners, "learner_weights": np.asarray(alphas),
            "classes": classes, "history": hist, "max_depth": max_depth,
            "requested_rounds": n_rounds, "actual_rounds": len(learners)}


def pred_samme(model, X):
    classes, alphas = model["classes"], model["learner_weights"]
    scores = np.zeros((len(X), len(classes)))
    for tree, alpha in zip(model["learners"], alphas):
        pred = tree.predict(X)
        for j, cls in enumerate(classes):
            scores[:, j] += alpha * (pred == cls)
    return classes[np.argmax(scores, axis=1)]


def save_boost_curves(model, depth, requested):
    h = model["history"]
    rounds = np.arange(1, model["actual_rounds"] + 1)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    series = [
        ("weighted_error", "Weighted training error", "Weighted error"),
        ("learner_weights", "Learner weight (alpha)", "Alpha"),
        ("max_sample_weight", "Maximum sample weight", "Maximum weight"),
        ("top10_weight", "Total weight of 10 heaviest images", "Combined weight"),
    ]
    for a, (key, title, ylabel) in zip(ax.flat, series):
        a.plot(rounds, h[key], marker="o", markersize=3)
        a.set_title(title); a.set_xlabel("Boosting round"); a.set_ylabel(ylabel); a.grid(True, alpha=.3)
    fig.suptitle(f"Selected SAMME model: depth={depth}, requested rounds={requested}")
    fig.tight_layout()
    path = FIGURES / "q2-b-boosting-curves.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("Saved", path)


def main():
    Xtr, ytr, Xv, yv, Xte, yte, seed = load_data()
    # Q2(b): select solely on clean validation accuracy.
    rows, models = [], {}
    for depth in DEPTHS:
        for rounds in ROUND_COUNTS:
            start = time.perf_counter()
            model = samme(Xtr, ytr, depth, rounds, seed)
            elapsed = time.perf_counter() - start
            train_acc = np.mean(pred_samme(model, Xtr) == ytr)
            val_acc = np.mean(pred_samme(model, Xv) == yv)
            models[(depth, rounds)] = model
            rows.append({"depth": depth, "requested_rounds": rounds,
                         "actual_rounds": model["actual_rounds"], "train_accuracy": train_acc,
                         "validation_accuracy": val_acc, "training_time_sec": elapsed})
    search = pd.DataFrame(rows)
    search.to_csv(RESULTS / "q2-b-samme-search.csv", index=False)
    print("\nQ2(b) SAMME search:\n", search.to_string(index=False))
    # Stable tie-breaking: first maximum, matching pandas idxmax in the notebook.
    best = search.loc[search["validation_accuracy"].idxmax()]
    best_depth, best_rounds = int(best["depth"]), int(best["requested_rounds"])
    selected = models[(best_depth, best_rounds)]
    print(f"\nSelected using validation accuracy only: depth={best_depth}, rounds={best_rounds}, "
          f"accepted={selected['actual_rounds']}, val={best['validation_accuracy']:.4f}")
    save_boost_curves(selected, best_depth, best_rounds)

    # Q2(c): compare training time/accuracy for 25, 50 and 100 learners.
    # Bagging trees use the same frozen Q1 tree setting.
    bag_rng = np.random.default_rng(seed)
    bag_trees, bag_time_at = [], {}
    start = time.perf_counter()
    for b in range(1, 101):
        idx = bag_rng.choice(len(Xtr), size=len(Xtr), replace=True)
        tree = DecisionTreeClassifier(max_depth=TREE_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                                      random_state=seed + b)
        tree.fit(Xtr[idx], ytr[idx]); bag_trees.append(tree)
        if b in B_VALUES: bag_time_at[b] = time.perf_counter() - start
    compare_rows = []
    boost_compare = {}
    for rounds in B_VALUES:
        start = time.perf_counter()
        bm = samme(Xtr, ytr, max_depth=3, n_rounds=rounds, seed=seed)
        bt = time.perf_counter() - start
        boost_compare[rounds] = bm
        for method, model, elapsed in [
            ("Bagging", bag_trees[:rounds], bag_time_at[rounds]),
            ("SAMME Boosting", bm, bt)]:
            if method == "Bagging":
                def bp(X):
                    pp = np.asarray([t.predict(X) for t in model])
                    return np.asarray([np.bincount(col.astype(int), minlength=40).argmax() for col in pp.T])
                ptrain, pval, ptest = bp(Xtr), bp(Xv), bp(Xte)
            else:
                ptrain, pval, ptest = pred_samme(model, Xtr), pred_samme(model, Xv), pred_samme(model, Xte)
            compare_rows.append({"method": method, "learners": rounds,
                "train_accuracy": np.mean(ptrain == ytr), "validation_accuracy": np.mean(pval == yv),
                "clean_test_accuracy": np.mean(ptest == yte), "training_time_sec": elapsed})
    comp = pd.DataFrame(compare_rows)
    comp.to_csv(RESULTS / "q2-c-bagging-vs-boosting.csv", index=False)
    print("\nQ2(c) bagging vs boosting:\n", comp.to_string(index=False))

    # Figure 2: compare train/validation/test accuracy and training time.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    metrics = [
        ("train_accuracy", "Training accuracy"),
        ("validation_accuracy", "Validation accuracy"),
        ("clean_test_accuracy", "Clean test accuracy"),
    ]
    x = np.arange(len(B_VALUES))
    width = 0.25
    for i, (column, label) in enumerate(metrics):
        for method, offset, marker in [("Bagging", -width / 2, "o"), ("SAMME Boosting", width / 2, "s")]:
            subset = comp[comp["method"] == method].set_index("learners").loc[B_VALUES]
            # Offset the two methods within each learner-count group.
            method_offset = (i - 1) * 0.0 + offset
            if i == 0:
                axes[0].bar(x + method_offset, subset[column].to_numpy(), width=width,
                            label=method if i == 0 else None, alpha=0.85)
        # Keep metric comparisons legible in a separate accuracy panel as lines.
    axes[0].clear()
    for method, marker in [("Bagging", "o"), ("SAMME Boosting", "s")]:
        subset = comp[comp["method"] == method].sort_values("learners")
        for column, label in metrics:
            axes[0].plot(subset["learners"], subset[column], marker=marker,
                         linestyle="-" if method == "Bagging" else "--",
                         label=f"{method}: {label}")
    axes[0].set_title("Accuracy comparison")
    axes[0].set_xlabel("Number of learners")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(B_VALUES)
    axes[0].grid(True, alpha=.3)
    axes[0].legend(fontsize=7)

    for method, marker in [("Bagging", "o"), ("SAMME Boosting", "s")]:
        subset = comp[comp["method"] == method].sort_values("learners")
        axes[1].plot(subset["learners"], subset["training_time_sec"], marker=marker,
                     linestyle="-" if method == "Bagging" else "--", label=method)
    axes[1].set_title("Training time comparison")
    axes[1].set_xlabel("Number of learners")
    axes[1].set_ylabel("Training time (seconds)")
    axes[1].set_xticks(B_VALUES)
    axes[1].grid(True, alpha=.3)
    axes[1].legend()
    fig.tight_layout()
    comparison_path = FIGURES / "q2-c-bagging-vs-boosting.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", comparison_path)

    # Inspect the training sample with the largest final weight in the 100-round depth-3 model.
    hard_model = boost_compare[100]
    weight_hist = np.asarray(hard_model["history"]["sample_weights_by_round"])
    if len(weight_hist):
        hard_idx = int(np.argmax(weight_hist[-1]))
        pred_hist = np.asarray(hard_model["history"]["training_predictions"])[:, hard_idx]
        hard_rows = pd.DataFrame({"round": np.arange(1, len(pred_hist)+1),
                                  "prediction": pred_hist,
                                  "true_identity": int(ytr[hard_idx]),
                                  "sample_weight": weight_hist[:, hard_idx]})
        hard_rows.to_csv(RESULTS / "q2-c-hardest-training-face-history.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        ax1.plot(hard_rows["round"], hard_rows["sample_weight"], marker=".", markersize=3)
        ax1.set_xlabel("Boosting round"); ax1.set_ylabel("Sample weight"); ax1.grid(True, alpha=.3)
        ax1.set_title(f"Hard training image (train position {hard_idx}, true class {ytr[hard_idx]})")
        fig.tight_layout(); path = FIGURES / "q2-c-hardest-training-face-weight.png"
        fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig); print("Saved", path)

        # Figure 4: the actual training face receiving the largest final SAMME weight.
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(Xtr[hard_idx].reshape(64, 64), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Highest-weight training face\nTrue identity: {int(ytr[hard_idx])}")
        ax.axis("off")
        fig.tight_layout()
        face_path = FIGURES / "q2-c-hardest-training-face.png"
        fig.savefig(face_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("Saved", face_path)

if __name__ == "__main__":
    main()
