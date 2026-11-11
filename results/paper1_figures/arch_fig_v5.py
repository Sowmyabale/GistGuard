# GXAI-Elderly Figure 1 (v5) — genuine flowchart: shapes, labeled data flows,
# swim-lanes per layer, offline silver-annotation loop. No paragraph boxes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle
import os

TEAL = "#2F8F8F"; TEAL_L = "#E3F1F0"; PUR = "#7A68B0"; PUR_L = "#EFEAF8"
CORAL = "#E4694E"; CORAL_L = "#FBEFE8"; NAVY = "#22344A"; GRAY = "#5B6B7A"
SLANT = 2.2

fig = plt.figure(figsize=(20, 11.5), dpi=110)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 220); ax.set_ylim(0, 112); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(110, 109.3, "GXAI-Elderly — System Flowchart", ha="center", va="center",
        fontsize=22, fontweight="bold", color=NAVY)
ax.text(110, 105.8, "data flow through the three layers; dashed = offline training/supervision",
        ha="center", va="center", fontsize=12.5, color=GRAY, style="italic")

# ---------- swim lanes ----------
def lane(y0, y1, color, label):
    ax.add_patch(FancyBboxPatch((0, y0), 220, y1 - y0, boxstyle="round,pad=0,rounding_size=0",
                                fc=color, ec="none", zorder=0))
    ax.text(3.2, (y0 + y1) / 2, label, rotation=90, ha="center", va="center",
            fontsize=13, fontweight="bold", color=GRAY)

lane(76, 102, TEAL_L, "LAYER 1 — DETECT")
lane(48, 74, PUR_L, "LAYER 2 — PROFILE")
lane(10, 46, CORAL_L, "LAYER 3 — EXPLAIN")

def box(x, y, w, h, text, fc, ec, fs=10.5, bold_first=True, lw=2.2, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.8",
                                fc="white", ec=ec, lw=lw, zorder=3))
    lines = text.split("\n")
    yy = y + h / 2 + (len(lines) - 1) * fs * 0.19
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, yy - i * fs * 0.38, ln, ha="center", va="center",
                fontsize=fs if i == 0 else fs - 1.1, fontweight="bold" if (i == 0 and bold_first) else "normal",
                color=tc if i == 0 else "#42525F", zorder=4)

def parallelogram(x, y, w, h, text, ec, fc, fs=10.5):
    ax.add_patch(Polygon([(x + SLANT, y), (x + w, y), (x + w - SLANT, y + h), (x, y + h)],
                         closed=True, fc=fc, ec=ec, lw=2.2, zorder=3))
    lines = text.split("\n")
    yy = y + h / 2 + (len(lines) - 1) * fs * 0.19
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, yy - i * fs * 0.38, ln, ha="center", va="center",
                fontsize=fs if i == 0 else fs - 1.2, fontweight="bold" if i == 0 else "normal",
                color=NAVY if i == 0 else "#42525F", zorder=4)

def cylinder(x, y, w, h, text, ec, fs=9.8):
    ry = min(2.2, h / 4)
    ax.add_patch(Polygon([(x, y + ry), (x, y + h - ry), (x + w, y + h - ry), (x + w, y + ry)],
                         closed=True, fc="white", ec=ec, lw=2, zorder=3))
    from matplotlib.patches import Ellipse
    ax.add_patch(Ellipse((x + w / 2, y + h - ry), w, ry * 2, fc="white", ec=ec, lw=2, zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + ry), w, ry * 2, fc="white", ec=ec, lw=2, zorder=3))
    lines = text.split("\n")
    yy = y + h / 2 + (len(lines) - 1) * fs * 0.19
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, yy - i * fs * 0.38, ln, ha="center", va="center",
                fontsize=fs if i == 0 else fs - 1.1, fontweight="bold" if i == 0 else "normal",
                color=NAVY if i == 0 else "#42525F", zorder=4)

def chip(x, y, w, h, text, ec, fs=9.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                                fc="#FFFFFF", ec=ec, lw=1.6, zorder=3))
    lines = text.split("\n")
    yy = y + h / 2 + (len(lines) - 1) * fs * 0.19
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, yy - i * fs * 0.38, ln, ha="center", va="center",
                fontsize=fs, color="#33424F", zorder=4)

def arrow(x0, y0, x1, y1, color=GRAY, dashed=False, label=None, lfs=8.6, rad=0.0, lw=2.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=20,
                                 lw=lw, color=color, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 1.6, label, fontsize=lfs, ha="center",
                color=color, style="italic", zorder=4,
                bbox=dict(fc="white", ec="none", pad=0.8))

# ---------- input ----------
parallelogram(8, 82, 34, 15, "SCAM CONVERSATION\nmulti-turn turns + victim config", TEAL, "#FFFFFF")
ax.text(25, 79.2, "calls · SMS · transcripts", fontsize=8.8, ha="center", color=GRAY, style="italic")

# ---------- LAYER 1 ----------
box(52, 83, 34, 13, "RoBERTa encoder\n(multi-task, shared)", TEAL, TEAL)
chip(94, 90, 40, 7.5, "scam-type head\n8 classes · 99.86% acc", TEAL)
chip(94, 80.5, 40, 7.5, "tactic head\n6 tactics · 99.86% acc", TEAL)
arrow(42, 89.5, 52, 89.5, label="turns")
arrow(86, 91, 94, 93.5); arrow(86, 88, 94, 84.5)

# offline annotation loop (top-right)
cylinder(150, 83, 46, 13, "LLM per-turn tactic annotation\n10,985 convs → silver v2 labels", PUR)
arrow(42, 96.5, 150, 96.5, dashed=True, label="conversations (offline)", rad=-0.12)
arrow(150, 87, 134, 84.5, dashed=True, rad=0.15)
ax.text(140, 78.6, "retrain tactic head", fontsize=8.6, ha="center", color=PUR, style="italic",
        bbox=dict(fc="white", ec="none", pad=0.8), zorder=4)

# ---------- LAYER 2 ----------
box(52, 55, 34, 13, "BiLSTM ToM profiler\n(susceptibility)", PUR, PUR)
chip(94, 56, 40, 11, "victim profile\ntrust · tech-savviness\nawareness — ~88% acc", PUR)
arrow(25, 82, 25, 61.5, color=GRAY); arrow(25, 61.5, 52, 61.5, label="turns")
arrow(86, 61.5, 94, 61.5)

# ---------- LAYER 3 ----------
box(52, 27, 34, 13, "evidence extractor\n(rule-based)", CORAL, CORAL)
chip(94, 27, 44, 13, "evidence card\nclaimed identity · requested items\npayment · pressure cues", CORAL)
arrow(25, 61.5, 25, 33.5, color=GRAY); arrow(25, 33.5, 52, 33.5, label="turns")
arrow(86, 33.5, 94, 33.5)

# silver dataset cylinder
cylinder(100, 13.5, 40, 9, "silver v2 gist dataset\n4,821 QC-filtered examples", NAVY, fs=9.2)

# generator
box(150, 24, 34, 22, "GIST GENERATOR\nLlama-3.1-8B + LoRA\nshort prompt + evidence card\nWARNING + ACTION", NAVY, NAVY, fs=10.2)
arrow(140, 61.5, 158, 46.5, label="profile", rad=-0.12)
arrow(138, 84, 156, 46.8, label="tactic", rad=-0.18)
arrow(138, 33.5, 150, 33.5, label="evidence card")
arrow(120, 22.5, 160, 24.5, dashed=True, label="SFT")

# output
parallelogram(190, 26, 26, 20, "GXAI GIST\nWARNING + ACTION", CORAL, "#FFFFFF", fs=10.2)
arrow(184, 35, 190, 35, label="gist")

# ---------- footer ----------
ax.add_patch(FancyBboxPatch((8, 2), 208, 5.4, boxstyle="round,pad=0.4,rounding_size=1.6",
                            fc="#F4F6F8", ec="#D7DEE4", lw=1.2))
ax.text(13, 4.7, "DATA:", fontsize=9.6, fontweight="bold", color=NAVY, va="center", ha="left")
ax.text(116, 4.7, "COVA-X 10,985 conversations (6,591 / 2,197 / 2,197, seed 42)   |   test gists 2,197   |   real-world: 243 YouTube scam-call transcripts",
        fontsize=9.2, color="#42525F", va="center", ha="center")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "framework_architecture_v5.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
