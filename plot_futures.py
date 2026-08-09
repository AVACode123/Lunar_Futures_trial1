from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# Settings
# ============================================================

INPUT_FILE = "runs/all_states.csv"
OUTPUT_DIR = "results"

PCA_FIGURE = os.path.join(
    OUTPUT_DIR,
    "lunar_futures_pca.png",
)

PCA_DATA_FILE = os.path.join(
    OUTPUT_DIR,
    "lunar_futures_pca.csv",
)

PCA_LOADINGS_FILE = os.path.join(
    OUTPUT_DIR,
    "lunar_futures_pca_loadings.csv",
)


# State variables used for PCA
FEATURES = [
    "us_china_tension",
    "shared_infrastructure",
    "scientific_openness",
    "mars_progress",
    "neutral_access",
    "us_power",
    "china_power",
    "us_public_cooperation",
    "china_public_cooperation",
    "third_force_strength",
    "international_trust",
]


# ============================================================
# Load simulation data
# ============================================================

def load_data() -> pd.DataFrame:
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}.\n"
            "Run batch_simulation.py first so that "
            "runs/all_states.csv is generated."
        )

    df = pd.read_csv(INPUT_FILE)

    missing = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following PCA variables are missing "
            f"from {INPUT_FILE}: {missing}"
        )

    if "run" not in df.columns:
        raise ValueError(
            'The input CSV must contain a "run" column.'
        )

    if "turn" not in df.columns:
        raise ValueError(
            'The input CSV must contain a "turn" column.'
        )

    if "event" not in df.columns:
        df["event"] = "none"

    return df


# ============================================================
# PCA
# ============================================================

def calculate_pca(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, PCA, pd.DataFrame]:

    # PCA should use only the numerical state variables.
    X = df[FEATURES].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(X_scaled)

    result = df.copy()

    result["PC1"] = coordinates[:, 0]
    result["PC2"] = coordinates[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=[
            "PC1_loading",
            "PC2_loading",
        ],
    )

    return result, pca, loadings


# ============================================================
# Plot
# ============================================================

def plot_worldlines(
    df: pd.DataFrame,
    pca: PCA,
) -> None:

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    plt.figure(
        figsize=(11, 8),
    )

    run_ids = sorted(
        df["run"].unique()
    )

    # Draw one trajectory for each simulated worldline.
    for run_id in run_ids:

        sub = (
            df[df["run"] == run_id]
            .sort_values("turn")
        )

        plt.plot(
            sub["PC1"],
            sub["PC2"],
            marker="o",
            linewidth=1.5,
            alpha=0.72,
            label=f"Run {run_id}",
        )

        # Mark the starting point.
        start = sub.iloc[0]

        plt.scatter(
            start["PC1"],
            start["PC2"],
            marker="s",
            s=45,
            zorder=4,
        )

        # Label the final point.
        end = sub.iloc[-1]

        plt.text(
            end["PC1"],
            end["PC2"],
            f"R{run_id}",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    # Mark turns in which an external event occurred.
    event_mask = ~df["event"].isin(
        [
            "none",
            "initial",
        ]
    )

    event_df = df[event_mask]

    if not event_df.empty:

        plt.scatter(
            event_df["PC1"],
            event_df["PC2"],
            marker="*",
            s=130,
            label="External event",
            zorder=6,
        )

    pc1_variance = (
        pca.explained_variance_ratio_[0]
        * 100
    )

    pc2_variance = (
        pca.explained_variance_ratio_[1]
        * 100
    )

    plt.xlabel(
        f"PC1 ({pc1_variance:.1f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({pc2_variance:.1f}% variance)"
    )

    plt.title(
        "Lunar Futures: "
        "PCA Trajectories of Simulated Worldlines"
    )

    plt.grid(
        alpha=0.2,
    )

    plt.legend(
        fontsize=8,
        ncol=2,
    )

    plt.tight_layout()

    plt.savefig(
        PCA_FIGURE,
        dpi=220,
    )

    plt.show()


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "\n=== Lunar Futures PCA Visualization ==="
    )

    df = load_data()

    print(
        f"Loaded {len(df)} state records "
        f"from {INPUT_FILE}"
    )

    result, pca, loadings = calculate_pca(
        df
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    result.to_csv(
        PCA_DATA_FILE,
        index=False,
    )

    loadings.to_csv(
        PCA_LOADINGS_FILE,
    )

    plot_worldlines(
        result,
        pca,
    )

    pc1 = (
        pca.explained_variance_ratio_[0]
        * 100
    )

    pc2 = (
        pca.explained_variance_ratio_[1]
        * 100
    )

    print(
        "\nPCA complete."
    )

    print(
        f"PC1 explained variance: {pc1:.1f}%"
    )

    print(
        f"PC2 explained variance: {pc2:.1f}%"
    )

    print(
        f"PC1 + PC2: {pc1 + pc2:.1f}%"
    )

    print(
        f"\nFigure saved to: {PCA_FIGURE}"
    )

    print(
        f"PCA coordinates saved to: "
        f"{PCA_DATA_FILE}"
    )

    print(
        f"PCA loadings saved to: "
        f"{PCA_LOADINGS_FILE}"
    )


if __name__ == "__main__":
    main()
