import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_mfi_heatmap(df: pd.DataFrame, title: str):

    """
    df columns:
      - critic
      - creator
      - value
    """

    pivot = df.pivot(
        index="critic",
        columns="creator",
        values="value",
    )
    def parse_mfi_targets(df):
        critics, creators = [], []
        for t in df["target_model"]:
            critic, creator = t.split("->")
            critics.append(critic.strip())
            creators.append(creator.strip())

        df["critic"] = critics
        df["creator"] = creators
        return df

    plt.figure(figsize=(10,6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=1.0)

    plt.title(title)
    plt.ylabel("Critic Model")
    plt.xlabel("Creator Model")
    plt.tight_layout()
    plt.show()