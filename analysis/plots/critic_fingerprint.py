import pandas as pd
import matplotlib.pyplot as plt


def plot_critic_fingerprint(df: pd.DataFrame, critic_name: str):
    """
    df must contain:
      - metric name
      - value
      - target_model
    """

    subset = df[df["target_model"] == critic_name]

    metrics = subset["name"].tolist()
    values = subset["value"].tolist()

    plt.figure(figsize=(6, 6))
    plt.bar(metrics, values)

    plt.title(f"Critic Fingerprint: {critic_name}")
    plt.ylabel("Metric Value")
    plt.tight_layout()
    plt.show()
