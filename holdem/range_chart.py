"""
range_chart.py
==============
Disegna le classiche griglie 13x13 delle "range" di poker (come la SB Open caricata)
e calcola la percentuale di mani giocate, pesata per il numero di combinazioni.

Concetti chiave:
- 169 "mani" preflop distinte: 13 coppie + 78 suited + 78 offsuit.
- Ma le COMBINAZIONI reali sono 1326: ogni coppia = 6 combo, ogni suited = 4, ogni
  offsuit = 12. La percentuale di una range si calcola SEMPRE pesando per le combo.
- Supporta FREQUENZE (es. "raise il 50% delle volte"): e' cosi' che i solver moderni
  esprimono le strategie GTO -> miste/probabilistiche. La cella appare rosa anziche'
  rossa piena. (Questo collega il tutto al tema della ricerca: la giocata ottimale
  e' una distribuzione di probabilita', non una scelta secca.)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RANKS = "AKQJT98765432"          # dall'alto al basso, come nelle chart
IDX = {r: i for i, r in enumerate(RANKS)}


def combos(r, c):
    """Numero di combinazioni della cella (riga r, colonna c) della griglia."""
    if r == c:      return 6     # coppia
    if r < c:       return 4     # suited (triangolo superiore)
    return 12                    # offsuit (triangolo inferiore)


def cell_label(r, c):
    """Etichetta come nella figura: 'AA', 'AK' (senza suffisso s/o, lo dice la posizione)."""
    hi, lo = RANKS[min(r, c)], RANKS[max(r, c)]
    return hi + hi if r == c else hi + lo


# --------------------------------------------------------------------------
#  Parser della notazione standard delle range
#  Esempi: "22+"  "A2s+"  "KTo+"  "T9s"  "QJo"  "55-22"  "A5s:0.5" (frequenza)
# --------------------------------------------------------------------------
def _add(cells, r, c, f):
    cells[(r, c)] = f


def parse_range(text):
    """Ritorna un dizionario {(riga, colonna): frequenza} dalle notazioni separate da virgola."""
    cells = {}
    for raw in text.replace(" ", "").split(","):
        if not raw:
            continue
        tok, _, fr = raw.partition(":")
        f = float(fr) if fr else 1.0

        if tok[:2] and tok[0] in IDX and tok[1] in IDX and tok[0] == tok[1]:
            # COPPIE: "NN", "NN+", "NN-MM"
            base = IDX[tok[0]]
            if tok.endswith("+"):
                for i in range(0, base + 1):       # da NN su fino ad AA
                    _add(cells, i, i, f)
            elif "-" in tok:
                hi = IDX[tok[0]]; lo = IDX[tok.split("-")[1][0]]
                for i in range(min(hi, lo), max(hi, lo) + 1):
                    _add(cells, i, i, f)
            else:
                _add(cells, base, base, f)
            continue

        # MANI NON-COPPIA: due rank + 's'/'o', opzionale '+'
        hi, lo = IDX[tok[0]], IDX[tok[1]]
        suited = "s" in tok
        plus = tok.endswith("+")
        # ordino: la cella suited e' (hi, lo) con hi<lo; offsuit (lo, hi)
        def put(h, l):
            if suited:  _add(cells, h, l, f)       # triangolo superiore
            else:       _add(cells, l, h, f)       # triangolo inferiore
        if plus:
            # '+' alza il kicker (lo) fino a una sotto la carta alta
            for l in range(lo, hi, -1):
                put(hi, l)
        else:
            put(hi, lo)
    return cells


def range_percent(cells):
    """Percentuale di combinazioni giocate (pesata per combo e per frequenza)."""
    played = sum(combos(r, c) * f for (r, c), f in cells.items())
    return 100.0 * played / 1326.0


# --------------------------------------------------------------------------
#  Disegno della griglia
# --------------------------------------------------------------------------
def freq_color(f):
    """Bianco (fold) -> rosso (raise). Le frequenze intermedie diventano rosa."""
    red = (0.85, 0.16, 0.16)
    return tuple(1 + f * (red[i] - 1) for i in range(3))


def draw(cells, title="Range", action="Raise", filename="range.png"):
    pct = range_percent(cells)
    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    for r in range(13):
        for c in range(13):
            f = cells.get((r, c), 0.0)
            ax.add_patch(Rectangle((c, 12 - r), 1, 1, facecolor=freq_color(f),
                                   edgecolor="black", linewidth=1.0))
            ax.text(c + 0.5, 12 - r + 0.5, cell_label(r, c),
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="#1a1a2e")
    ax.set_xlim(0, 13); ax.set_ylim(-1.2, 14)
    ax.axis("off"); ax.set_aspect("equal")
    ax.text(0, 13.4, title, fontsize=20, fontweight="bold", color="#1a1a2e")
    ax.text(13, 13.4, f"{pct:.1f}%", fontsize=20, fontweight="bold",
            color="#c92a2a", ha="right")
    ax.add_patch(Rectangle((0, -1.0), 0.7, 0.7, facecolor=freq_color(1.0),
                           edgecolor="black"))
    ax.text(0.9, -0.65, action, fontsize=12, va="center")
    plt.tight_layout()
    plt.savefig(filename, dpi=130, bbox_inches="tight")
    plt.close()
    return pct


if __name__ == "__main__":
    # Range "SB Open" trascritta dalla figura caricata
    sb_open = ("22+, A2s+, K2s+, Q3s+, J5s+, T6s+, 96s+, 86s+, 75s+, 65s, 54s, "
               "A4o+, K8o+, Q9o+, J9o+, T8o+, 98o")
    cells = parse_range(sb_open)
    pct = draw(cells, title="SB Open", action="Raise", filename="sb_open.png")
    print(f"SB Open -> {pct:.1f}%  ({len(cells)} mani distinte)")

    # Esempio con STRATEGIA MISTA (alcune mani raise solo a una certa frequenza)
    mixed = parse_range("99+, AQs+, AKo, AJs:0.5, KQs:0.5, A5s:0.5, 77:0.3")
    draw(mixed, title="Esempio GTO (misto)", action="Raise (freq.)",
         filename="mixed_example.png")
    print(f"Esempio misto -> {range_percent(parse_range('99+, AQs+, AKo')):.1f}% (solo le pure)")
