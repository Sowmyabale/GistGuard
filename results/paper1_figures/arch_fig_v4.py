# GXAI-Elderly catchy architecture figure (v4) — fixed layout with imshow illustrations.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from PIL import Image
import os

WS = os.path.dirname(os.path.abspath(__file__))
elderly = Image.open(os.path.join(WS, "illus_elderly.png"))
scammer = Image.open(os.path.join(WS, "illus_scammer.png"))

TEAL = "#3A8B8B"; CORAL = "#E4694E"; NAVY = "#22344A"; CREAM = "#FDF6EE"
LIGHT = "#EAF4F3"; GRAY = "#5B6B7A"

fig = plt.figure(figsize=(20, 12.5), dpi=110)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 200); ax.set_ylim(0, 125); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(100, 121.5, "GXAI-Elderly: Grounded Gist Explanations for Scam Protection",
        ha="center", va="center", fontsize=22, fontweight="bold", color=NAVY)
ax.text(100, 117.3, "a three-layer, gist-based explainable AI framework for older adults",
        ha="center", va="center", fontsize=13, color=GRAY, style="italic")

# ---------- left scene (illustrations via imshow) ----------
ax.imshow(scammer, extent=(1, 33, 88, 120), zorder=2, aspect="auto", interpolation="antialiased")
bub = FancyBboxPatch((35, 93), 46, 17, boxstyle="round,pad=1.2,rounding_size=3",
                     fc="white", ec=CORAL, lw=2)
ax.add_patch(bub)
ax.text(58, 101.5, T if False else "\"This is your bank's Fraud\nDepartment. Verify your card\nnumber and security code NOW!\"",
        ha="center", va="center", fontsize=10, color=NAVY)

ax.text(17, 76, "\u26A0", fontsize=40, ha="center", va="center", color="#D9534F")
ax.text(17, 70.5, "suspicious call", fontsize=10, ha="center", color="#D9534F", fontweight="bold")

ax.imshow(elderly, extent=(1, 36, 15, 55), zorder=2, aspect="auto", interpolation="antialiased")
ax.text(19, 11, "victim receives call / SMS", fontsize=10, ha="center", color=GRAY, style="italic")

ax.add_patch(FancyArrowPatch((82, 63), (97, 63), arrowstyle="-|>", mutation_scale=28, lw=4, color=GRAY))

# ---------- middle pipeline ----------
def layer_box(x, y, w, h, title, lines, fc, ec):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=1,rounding_size=2.6", fc=fc, ec=ec, lw=2.4)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h - 4.0, title, ha="center", va="center", fontsize=13, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, y + (h - 7.5) / 2, lines, ha="center", va="center", fontsize=10.4, color="#33424F", linespacing=1.55)

PX, PW = 100, 50
layer_box(PX, 89, PW, 19, "LAYER 1 — MULTI-TASK DETECTION",
          "RoBERTa\nscam type (8 classes) · dominant tactic\naccuracy 99.86%", LIGHT, TEAL)
layer_box(PX, 63, PW, 18, "LAYER 2 — ToM PROFILER",
          "BiLSTM susceptibility profile\ntrust · tech-savviness · awareness\naccuracy ~88%", LIGHT, TEAL)
layer_box(PX, 41, PW, 15, "EVIDENCE EXTRACTOR",
          "rule-based evidence card\nclaimed identity · requested items · pressure", "#FBF3E4", CORAL)
layer_box(PX, 13, PW, 20, "LAYER 3 — GIST GENERATOR",
          "Llama-3.1-8B + LoRA (silver v2)\nWARNING: what + why it's a scam\nACTION: one protective step", LIGHT, TEAL)

for y0 in (87, 61, 39):
    ax.add_patch(FancyArrowPatch((PX + PW / 2, y0), (PX + PW / 2, y0 - 3.5),
                                 arrowstyle="-|>", mutation_scale=22, lw=3, color=GRAY))

# coral side arrow: tactic + profile feed into L3
ax.annotate("", xy=(PX + PW + 2, 24), xytext=(PX + PW + 2, 72),
            arrowprops=dict(arrowstyle="-|>", mutation_scale=22, lw=3, color=CORAL,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(PX + PW + 8.5, 49, "profile + tactic", fontsize=10, color=CORAL,
        rotation=90, ha="center", va="center")

ax.add_patch(FancyArrowPatch((PX + PW + 13, 44), (168, 44), arrowstyle="-|>", mutation_scale=28, lw=4, color=GRAY))

# ---------- right output panel ----------
OX, OY, OW, OH = 170, 24, 28, 66
ax.add_patch(FancyBboxPatch((OX, OY), OW, OH, boxstyle="round,pad=1,rounding_size=2.8", fc=NAVY, ec=NAVY))
for i, c in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
    ax.add_patch(Circle((OX + 2.6 + i * 2.9, OY + OH - 3.2), 1.0, fc=c, ec="none"))
ax.text(OX + OW / 2 + 4.5, OY + OH - 3.2, "GXAI OUTPUT", ha="center", va="center",
        fontsize=11.5, fontweight="bold", color="white")

ax.text(OX + 2.2, OY + OH - 10, '"warning":', fontsize=11, color="#7FD1C0", fontweight="bold", va="top")
ax.text(OX + 2.2, OY + OH - 14, '"This is not a real bank', fontsize=10, color="white", va="top")
ax.text(OX + 2.2, OY + OH - 17.6, 'employee. They\'re trying', fontsize=10, color="white", va="top")
ax.text(OX + 2.2, OY + OH - 21.2, 'to trick you into giving', fontsize=10, color="white", va="top")
ax.text(OX + 2.2, OY + OH - 24.8, 'them your card number and', fontsize=10, color="white", va="top")
ax.text(OX + 2.2, OY + OH - 28.4, 'security code."', fontsize=10, color="white", va="top")
ax.text(OX + 2.2, OY + OH - 34.5, '"action":', fontsize=11, color="#F4C08A", fontweight="bold", va="top")
ax.text(OX + 2.2, OY + OH - 38, '"Hang up. Call the number', fontsize=10, color="white", va="top")
ax.text(OX + 2.2, OY + OH - 41.6, 'on the back of your card."', fontsize=10, color="white", va="top")

def badge(y, txt):
    ax.add_patch(FancyBboxPatch((OX + 2.2, y), OW - 4.4, 5.6, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc="#2E4258", ec="#456078", lw=1))
    ax.text(OX + OW / 2, y + 2.8, txt, ha="center", va="center", fontsize=9.3, color="#BFE3DC", fontweight="bold")

badge(OY + 11.5, "FK grade 7.0 · 23 words")
badge(OY + 4, "ask-coverage 87% · grounded")

# ---------- bottom steps strip ----------
sy = 1.2
steps = [("1  INPUT", "scam call or SMS\nconversation"),
         ("2  ANALYZE", "detect scam type, tactic,\nprofile the victim"),
         ("3  PROTECT", "gist warning + one\nprotective action")]
sw = 56
for i, (t, s) in enumerate(steps):
    x = 58 + i * 46
    ax.add_patch(FancyBboxPatch((x, sy), sw - 8, 8.6, boxstyle="round,pad=0.8,rounding_size=2.2",
                                fc=CREAM if i != 1 else LIGHT, ec=CORAL if i == 2 else TEAL, lw=2))
    ax.text(x + 5.5, sy + 5.6, t, fontsize=12, fontweight="bold", color=NAVY, va="center")
    ax.text(x + 5.5, sy + 2.4, s, fontsize=9.4, color=GRAY, va="center", linespacing=1.4)
    if i < 2:
        ax.add_patch(FancyArrowPatch((x + sw - 6.5, sy + 4.3), (x + sw - 3.5, sy + 4.3),
                                     arrowstyle="-|>", mutation_scale=18, lw=3, color=GRAY))

out = os.path.join(WS, "framework_architecture_v4.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("saved", out)
