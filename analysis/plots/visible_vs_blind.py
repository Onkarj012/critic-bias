import matplotlib.pyplot as plt
import pandas as pd


def plot_visible_blind_delta(df_visible, df_blind):
    """
    Both dataframes indexed by target_model
    """

    merged = df_visible.merge(
        df_blind,
        on="target_model",
        suffixes=("_visible", "_blind"),
    )

    merged["delta"] = merged["value_visible"] - merged["value_blind"]

    merged = merged.sort_values("delta")

    plt.figure(figsize=(10, 5))
    plt.barh(merged["target_model"], merged["delta"])
    plt.axvline(0, linestyle="--")

    plt.title("Visible vs Blind Bias Delta (Branding Effect)")
    plt.xlabel("Score Difference")
    plt.tight_layout()
    plt.show()
