from pathlib import Path
import sys
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

setup_plot()
fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 120); ax.set_ylim(0, 68); ax.axis("off")

C_HEAD = "#37474F"; C_BOX = "#EEF3F6"; C_ACC = "#DCE9DC"; C_WARN = "#F5E6D3"; C_OUT = "#E3EDF5"

def box(x, y, w, h, title, lines, fc=C_BOX, title_fs=11, fs=9.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=C_HEAD, lw=1.3))
    ax.text(x + w/2, y + h - 2.8, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color=C_HEAD)
    for i, ln in enumerate(lines):
        ax.text(x + w/2, y + h - 6.0 - i*3.0, ln, ha="center", va="center",
                fontsize=fs, color="#263238")

def arrow(p1, p2, label=None, lx=0, ly=0, dashed=False):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.6,
                                 color="#546E7A", linestyle="--" if dashed else "-"))
    if label:
        ax.text((p1[0]+p2[0])/2 + lx, (p1[1]+p2[1])/2 + ly, label, fontsize=8.4,
                ha="center", color="#546E7A", style="italic")

# Main horizontal flow (y=22..44)
box(1, 22, 13, 22, "INPUT", ["scam conversation", "(multi-turn call /", "SMS text)"], fc=C_OUT)
box(18, 22, 20, 22, "LAYER 1", ["Multi-Task RoBERTa", "scam type (8 classes)", "dominant tactic (6)", "acc 99.86%"])
box(64, 22, 30, 22, "LAYER 3", ["Evidence-Grounded Gist Generator", "silver v2 (4,821) + QC v2", "LoRA SFT (Llama-3.1-8B)", "ask-cov 87% | FK 7.0 | uniq 60%"])
box(97, 22, 21, 22, "OUTPUT", ["WARNING: claimed identity", "+ requested item + why", "it is a scam", "ACTION: one step"], fc=C_OUT, title_fs=10.5, fs=8.8)

# Bottom row
box(18, 2, 20, 16, "LAYER 2", ["ToM Susceptibility Profiler", "(BiLSTM) trust / tech /", "awareness | acc ~88%"], title_fs=10.5, fs=8.8)
box(41, 2, 20, 16, "EVIDENCE", ["EXTRACTOR (rule-based)", "claimed identity | requested", "items | payment | pressure"], fc=C_ACC, title_fs=10.5, fs=8.8)

# Top offline annotation
box(41, 50, 24, 14, "LLM TACTIC ANNOTATION", ["(offline, per-turn) 10,985 convs", "mapping agreement only 52.8%"], fc=C_WARN, title_fs=10.5, fs=8.8)

# Main arrows
arrow((14.5, 33), (18, 33))
arrow((38, 33), (64, 33), label="tactic", lx=-1.2, ly=1.6)
arrow((94, 33), (97, 33))
# L2 -> L3 (elbow, avoids evidence box: right at y=10 then up through gap x=58..62)
ax.plot([38, 58], [10, 10], color="#546E7A", lw=1.6)
ax.plot([58, 58], [10, 21], color="#546E7A", lw=1.6)
arrow((58, 21), (63.4, 22.2), label="susceptibility profile", lx=4.2, ly=-3.4)
# Evidence -> L3
arrow((61, 18), (70, 22), label="evidence card", lx=2.6, ly=-2.4)
# Annotation dashed feeds
arrow((47, 50), (31, 44.4), label="retrain tactic head", lx=-4.5, ly=1.6, dashed=True)
arrow((58, 50), (76, 44.4), label="real tactic labels (silver v2)", lx=4, ly=1.8, dashed=True)

ax.set_title("GXAI-Elderly: Three-Layer Framework with Evidence-Grounded Gist Generation",
             fontsize=13.5, fontweight="bold", color=C_HEAD, pad=12)
fig.savefig(r"C:\Users\Sowmya\Desktop\Research in Aging\paper1_figures\framework_architecture_v3.png", dpi=220, bbox_inches="tight")
plt.close(fig)
print("saved")
