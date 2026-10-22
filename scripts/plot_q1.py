#!/usr/bin/env python3
"""
Reproduce the Q1 figures from answer_1.ipynb.

Run from the repository root:
    python scripts/plot_q1.py

Outputs (saved under figures/):
    q1-c-ensemble-diagnostics.png
    q1-d-bagging-corrects-tree.png
    q1-d-both-models-wrong.png

The script recreates the Q1 pipeline because the figures depend on trained models,
predictions, bootstrap samples, OOB masks, and test-case selections. It reuses
olivetti_split.npz if it exists; otherwise it creates and saves the deterministic
split using the numeric digits of ROLL_NO.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works without a GUI/display.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier


# Repository paths
ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
SPLIT_PATH = ROOT / "olivetti_split.npz"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Keep this identical to the notebook's roll number.
ROLL_NO = "M26CSA016"
B_VALUES = [1, 5, 10, 25, 50, 100, 200]
N_IDENTITIES = 40


def load_data_and_split():
    """Load Olivetti faces and reuse the saved split, creating it if absent."""
    faces = fetch_olivetti_faces()
    X = faces.images.reshape(len(faces.images), -1)
    y = faces.target

    if SPLIT_PATH.exists():
        saved = np.load(SPLIT_PATH)
        train_idx = saved["train_indices"]
        val_idx = saved["val_indices"]
        test_idx = saved["test_indices"]
        # Do not silently use a split for a different dataset size.
        if (len(train_idx) + len(val_idx) + len(test_idx)) != len(y):
            raise ValueError(
                f"{SPLIT_PATH} does not match the loaded dataset. "
                "Check the saved split file."
            )
        print(f"Reusing saved split: {SPLIT_PATH}")
    else:
        seed_digits = "".join(ch for ch in ROLL_NO if ch.isdigit())
        if not seed_digits:
            raise ValueError("ROLL_NO must contain at least one digit.")
        seed = int(seed_digits)
        rng = np.random.default_rng(seed)

        train_idx, val_idx, test_idx = [], [], []
        for identity in np.unique(y):
            identity_indices = np.where(y == identity)[0]
            shuffled = rng.permutation(identity_indices)
            train_idx.extend(shuffled[:6])
            val_idx.extend(shuffled[6:8])
            test_idx.extend(shuffled[8:10])

        train_idx = np.asarray(train_idx, dtype=int)
        val_idx = np.asarray(val_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        np.savez(
            SPLIT_PATH,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            seed=seed,
        )
        print(f"Created and saved deterministic split: {SPLIT_PATH}")

    # Seed used by the notebook for tree/bootstrap randomness.
    seed_digits = "".join(ch for ch in ROLL_NO if ch.isdigit())
    seed = int(seed_digits)
    return (
        X, y, train_idx, val_idx, test_idx,
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx], seed,
    )


def select_baseline_tree(X_train, y_train, X_val, y_val, seed):
    """Replicate the notebook's 15-setting validation search."""
    depths = [2, 4, 8, 16, None]
    leaves = [1, 2, 4]
    best_model = None
    best_val_accuracy = -1.0
    best_depth = None
    best_min_leaf = None

    for depth in depths:
        for min_leaf in leaves:
            model = DecisionTreeClassifier(
                max_depth=depth,
                min_samples_leaf=min_leaf,
                random_state=seed,
            )
            model.fit(X_train, y_train)
            val_accuracy = model.score(X_val, y_val)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = model
                best_depth = depth
                best_min_leaf = min_leaf

    print(
        f"Selected baseline tree: max_depth={best_depth}, "
        f"min_samples_leaf={best_min_leaf}, "
        f"validation_accuracy={best_val_accuracy:.4f}"
    )
    return best_model, best_depth, best_min_leaf


def majority_vote(predictions):
    final_predictions = []
    for column in predictions.T:
        counts = np.bincount(column.astype(int), minlength=N_IDENTITIES)
        final_predictions.append(np.argmax(counts))
    return np.asarray(final_predictions)


def train_bagging(X_train, y_train, X_val, X_test, best_depth, best_min_leaf, seed):
    """Replicate the notebook's 200 bootstrap trees and predictions."""
    n_train = len(X_train)
    bootstrap_rng = np.random.default_rng(seed)
    trees, bootstrap_indices_list, distinct_counts = [], [], []

    for b in range(200):
        bootstrap_indices = bootstrap_rng.choice(
            n_train, size=n_train, replace=True
        )
        bootstrap_indices_list.append(bootstrap_indices)
        distinct_counts.append(len(np.unique(bootstrap_indices)))

        tree = DecisionTreeClassifier(
            max_depth=best_depth,
            min_samples_leaf=best_min_leaf,
            random_state=seed + b,
        )
        tree.fit(X_train[bootstrap_indices], y_train[bootstrap_indices])
        trees.append(tree)

    train_predictions = np.asarray([tree.predict(X_train) for tree in trees])
    val_predictions = np.asarray([tree.predict(X_val) for tree in trees])
    test_predictions = np.asarray([tree.predict(X_test) for tree in trees])
    return (
        trees, bootstrap_indices_list, distinct_counts,
        train_predictions, val_predictions, test_predictions,
    )


def save_diagnostics_plot(B_values, results_c_df):
    """Save the notebook's three-panel OOB/disagreement/vote-margin figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(results_c_df["B"], results_c_df["OOB_accuracy"], marker="o")
    axes[0].set_title("OOB Accuracy")
    axes[0].set_xlabel("Number of Trees (B)")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)

    axes[1].plot(
        results_c_df["B"], results_c_df["pairwise_disagreement"], marker="o"
    )
    axes[1].set_title("Pairwise Disagreement")
    axes[1].set_xlabel("Number of Trees (B)")
    axes[1].set_ylabel("Disagreement")
    axes[1].grid(True)

    axes[2].plot(
        results_c_df["B"], results_c_df["mean_vote_margin"], marker="o"
    )
    axes[2].set_title("Mean Vote Margin")
    axes[2].set_xlabel("Number of Trees (B)")
    axes[2].set_ylabel("Vote Margin")
    axes[2].grid(True)

    fig.tight_layout()
    path = FIGURES_DIR / "q1-c-ensemble-diagnostics.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def evaluate_models(X_eval, y_true, best_model, final_trees, B_final):
    tree_predictions = best_model.predict(X_eval)
    tree_probabilities = best_model.predict_proba(X_eval)
    tree_confidence = np.max(tree_probabilities, axis=1)

    bagging_votes = np.asarray([tree.predict(X_eval) for tree in final_trees])
    bagging_predictions, bagging_confidence, all_vote_counts = [], [], []

    for image_predictions in bagging_votes.T:
        vote_counts = np.bincount(
            image_predictions.astype(int), minlength=N_IDENTITIES
        )
        bagging_predictions.append(np.argmax(vote_counts))
        bagging_confidence.append(np.max(vote_counts) / B_final)
        all_vote_counts.append(vote_counts)

    return (
        {"Tree accuracy": np.mean(tree_predictions == y_true),
         "Tree mean confidence": np.mean(tree_confidence),
         "Bagging accuracy": np.mean(np.asarray(bagging_predictions) == y_true),
         "Bagging mean confidence": np.mean(bagging_confidence)},
        tree_predictions,
        tree_confidence,
        np.asarray(bagging_predictions),
        np.asarray(bagging_confidence),
        np.asarray(all_vote_counts),
    )


def save_case_figure(indices, title, filename, X_test_clean, y_test, tree_pred, bag_pred):
    selected = indices[:2]
    if len(selected) == 0:
        print(f"Skipping {filename}: no examples available for '{title}'.")
        return

    fig, axes = plt.subplots(1, len(selected), figsize=(4 * len(selected), 4))
    axes = np.atleast_1d(axes)

    for ax, i in zip(axes, selected):
        ax.imshow(X_test_clean[i].reshape(64, 64), cmap="gray")
        ax.set_title(f"True: {y_test[i]}\nTree: {tree_pred[i]}, Bag: {bag_pred[i]}")
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    (
        X, y, train_idx, val_idx, test_idx,
        X_train, y_train, X_val, y_val, X_test, y_test, seed,
    ) = load_data_and_split()

    best_model, best_depth, best_min_leaf = select_baseline_tree(
        X_train, y_train, X_val, y_val, seed
    )

    (
        bagged_trees, bootstrap_indices_list, distinct_counts,
        train_predictions, val_predictions, test_predictions,
    ) = train_bagging(
        X_train, y_train, X_val, X_test, best_depth, best_min_leaf, seed
    )

    # Recreate Q1(c): OOB accuracy, pairwise disagreement, mean vote margin.
    n_train = len(X_train)
    oob_masks = []
    for bootstrap_indices in bootstrap_indices_list:
        mask = np.ones(n_train, dtype=bool)
        mask[np.unique(bootstrap_indices)] = False
        oob_masks.append(mask)
    oob_masks = np.asarray(oob_masks)

    results_c = []
    for B in B_VALUES:
        oob_vote_counts = np.zeros((n_train, N_IDENTITIES), dtype=int)
        for t in range(B):
            oob_indices = np.where(oob_masks[t])[0]
            oob_labels = train_predictions[t, oob_indices].astype(int)
            oob_vote_counts[oob_indices, oob_labels] += 1

        has_oob_votes = oob_vote_counts.sum(axis=1) > 0
        oob_predictions = np.argmax(oob_vote_counts, axis=1)
        oob_accuracy = np.mean(
            oob_predictions[has_oob_votes] == y_train[has_oob_votes]
        )

        disagreements = []
        for i in range(B):
            for j in range(i + 1, B):
                disagreements.append(np.mean(val_predictions[i] != val_predictions[j]))
        avg_disagreement = np.mean(disagreements) if disagreements else np.nan

        margins = []
        for image_pred in val_predictions[:B].T:
            vote_counts = np.bincount(image_pred.astype(int), minlength=N_IDENTITIES)
            sorted_votes = np.sort(vote_counts)
            margins.append((sorted_votes[-1] / B) - (sorted_votes[-2] / B))

        results_c.append({
            "B": B,
            "OOB_accuracy": oob_accuracy,
            "pairwise_disagreement": avg_disagreement,
            "mean_vote_margin": np.mean(margins),
        })

    results_c_df = pd.DataFrame(results_c)
    print("\nQ1(c): ENSEMBLE DIAGNOSTICS")
    print(results_c_df.to_string(index=False))
    save_diagnostics_plot(B_VALUES, results_c_df)

    # Recreate Q1(d): clean and central-16x16-masked test cases.
    B_final = 50
    final_trees = bagged_trees[:B_final]
    X_test_clean = X_test.copy()
    masked_images = X_test.reshape(-1, 64, 64).copy()
    masked_images[:, 24:40, 24:40] = 0
    X_test_masked = masked_images.reshape(len(X_test), -1)

    clean_result = evaluate_models(X_test_clean, y_test, best_model, final_trees, B_final)
    masked_result = evaluate_models(X_test_masked, y_test, best_model, final_trees, B_final)
    comparison_df = pd.DataFrame([
        {"Test set": "Clean", **clean_result[0]},
        {"Test set": "Masked", **masked_result[0]},
    ])
    print("\nQ1(d): ACCURACY AND MEAN PREDICTION CONFIDENCE")
    print(comparison_df.to_string(index=False))

    (_, tree_pred, _, bag_pred, _, vote_counts) = clean_result
    corrected = np.where((tree_pred != y_test) & (bag_pred == y_test))[0]
    both_wrong = np.where((tree_pred != y_test) & (bag_pred != y_test))[0]

    save_case_figure(
        corrected,
        "Bagging Corrects the Tree",
        "q1-d-bagging-corrects-tree.png",
        X_test_clean, y_test, tree_pred, bag_pred,
    )
    save_case_figure(
        both_wrong,
        "Both Models Predict Incorrectly",
        "q1-d-both-models-wrong.png",
        X_test_clean, y_test, tree_pred, bag_pred,
    )


if __name__ == "__main__":
    main()
