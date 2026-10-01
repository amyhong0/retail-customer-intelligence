"""저장된 지표를 사용해 모델 재학습 없이 그래프를 생성합니다."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .features import feature_dictionary


def render_plots(rec_metrics: pd.DataFrame, importance: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    (rec_metrics.set_index("model")[["precision@10", "recall@10", "ndcg@10"]]
     .rename(columns={"precision@10": "정밀도@10", "recall@10": "재현율@10", "ndcg@10": "NDCG@10"})
     .plot(kind="bar", ax=ax, rot=0))
    ax.set(title="추천 모델 성능 비교", ylabel="지표 값", xlabel="")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "recommendation_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    dictionary = feature_dictionary()
    labels = dict(zip(dictionary.feature, dictionary.definition))
    plot_importance = importance.head(10).copy()
    plot_importance["feature"] = plot_importance.feature.map(labels).fillna(plot_importance.feature)
    sns.barplot(data=plot_importance, x="importance", y="feature", ax=ax, color="#0f766e")
    ax.set(title="구매 예측에 중요했던 고객 특성", xlabel="특성 값을 섞었을 때의 성능 감소", ylabel="")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
