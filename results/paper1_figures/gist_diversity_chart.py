from pathlib import Path
import sys

sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

setup_plot()

scams = ["bank", "gov_impers.", "grandparent", "investment", "lottery", "medicare", "romance", "virt_kidnap"]
rows = []
for variant, vals in [
    ("v1.5 Llama (prompted, old)", [78, 74, 91, 100, 100, 73, 83, 86]),
    ("v3 prompted + QC",           [74, 82, 80, 83, 92, 48, 83, 52]),
    ("v3 LoRA (fine-tuned)",       [42, 74, 37, 57, 46, 21, 47, 13]),
]:
    for s, v in zip(scams, vals):
        rows.append({"Scam type": s, "Variant": variant, "Unique warnings (%)": v})

df = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df, x="Scam type", y="Unique warnings (%)", hue="Variant", ax=ax)
ax.set_title("Gist diversity: % unique warnings per scam type (n = test split, 2197)")
ax.set_xlabel("")
ax.set_ylabel("Unique warnings (%)")
ax.axhline(50, ls="--", lw=1, color="gray")
sns.move_legend(ax, "upper right", title="")
fig.savefig(r"C:\Users\Sowmya\Desktop\Research in Aging\paper1_figures\gist_diversity.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("saved")
