"""
multi_run.py
============
Il self-play e' un processo RUMOROSO: ogni esecuzione, partendo da un seme casuale
diverso, segue una traiettoria un po' diversa. Per una ricerca seria non basta UNA
run: si ripete l'esperimento piu' volte e si mostra la MEDIA con la fascia di
variabilita' (deviazione standard).

Genera: convergenza_multi.png  (curva media + banda min/max sui diversi semi).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kuhn, nfsp, train


def one_run(seed, episodes, eval_every):
    # reimposto lo stesso generatore casuale ovunque, per controllare il seme
    nfsp.reseed(seed)
    train.rng = nfsp.rng
    agents = [nfsp.NFSPAgent(), nfsp.NFSPAgent()]
    xs, ys = [], []
    for ep in range(1, episodes + 1):
        train.play_episode(agents)
        agents[0].learn(); agents[1].learn()
        if ep % eval_every == 0:
            s = train.current_average_strategy(agents)
            xs.append(ep); ys.append(kuhn.exploitability(s, s))
    return xs, ys


def main(seeds=(0, 1, 2), episodes=40_000, eval_every=4_000):
    curves = []
    xs = None
    for sd in seeds:
        xs, ys = one_run(sd, episodes, eval_every)
        curves.append(ys)
        print(f"  seme {sd}: exploitability finale = {ys[-1]:.4f}")
    Y = np.array(curves)
    mean, lo, hi = Y.mean(0), Y.min(0), Y.max(0)

    plt.figure(figsize=(8, 5))
    plt.fill_between(xs, lo, hi, alpha=0.2, color="#2563eb", label="min-max tra i semi")
    plt.plot(xs, mean, lw=2.2, color="#1d4ed8", label="media")
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    plt.xlabel("Partite di self-play")
    plt.ylabel("Exploitability  (0 = Nash)")
    plt.title(f"Convergenza media su {len(seeds)} esecuzioni indipendenti")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("convergenza_multi.png", dpi=130)
    print("Grafico salvato: convergenza_multi.png")


if __name__ == "__main__":
    main()
