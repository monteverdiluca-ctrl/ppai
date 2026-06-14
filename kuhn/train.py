"""
train.py
========
Fa giocare le due reti NFSP l'una contro l'altra (SELF-PLAY), le allena, e misura
ogni tot partite quanto la STRATEGIA MEDIA appresa sia vicina all'equilibrio di Nash
(metrica = exploitability, calcolata in modo esatto da kuhn.py).

Output:
  - grafico  convergenza.png   (exploitability che scende verso 0)
  - tabella di confronto tra la strategia imparata dalla rete e il Nash esatto (CFR)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kuhn
from cfr import train as cfr_train
from nfsp import NFSPAgent, rng

CARD = {0: "J", 1: "Q", 2: "K"}


def play_episode(agents):
    """Una mano di self-play. Ritorna i dati raccolti per RL e SL."""
    deal = kuhn.DEALS[rng.integers(len(kuhn.DEALS))]
    # ogni giocatore decide se questa mano la gioca da 'best response' o da 'media'
    br_mode = [rng.random() < agents[p].eta for p in (0, 1)]

    history = ""
    taken = [[], []]                       # mosse (x, azione) per giocatore
    while not kuhn.is_terminal(history):
        p = kuhn.player_to_act(history)
        a, x = agents[p].act(deal[p], history, br_mode[p])
        taken[p].append((x, a))
        history += "b" if a == 1 else "p"

    u0 = kuhn.payoff_p0(deal, history)
    returns = [u0, -u0]                     # gioco a somma zero

    for p in (0, 1):
        if br_mode[p]:                      # raccolgo dati solo dalle mani 'best response'
            for x, a in taken[p]:
                agents[p].rl_mem.add((x, a, returns[p]))   # RL: (stato, azione, ritorno)
                agents[p].sl_mem.add((x, a))               # SL: (stato, azione)


def current_average_strategy(agents):
    """Estrae dalla rete Pi la prob. di puntare in OGNI situazione di gioco."""
    s = {}
    for c in range(3):
        for h in kuhn.HISTORIES:
            p = kuhn.player_to_act(h)
            s[kuhn.infoset_key(c, h)] = agents[p].average_policy_bet_prob(c, h)
    return s


def run(episodes=90_000, eval_every=3_000):
    agents = [NFSPAgent(), NFSPAgent()]
    xs, ys = [], []
    for ep in range(1, episodes + 1):
        play_episode(agents)
        agents[0].learn(); agents[1].learn()
        if ep % eval_every == 0:
            s = current_average_strategy(agents)
            expl = kuhn.exploitability(s, s)
            xs.append(ep); ys.append(expl)
            print(f"  partita {ep:>7,}   exploitability = {expl:.4f}")
    return agents, xs, ys


if __name__ == "__main__":
    print("Equilibrio esatto di riferimento (CFR)...")
    nash = cfr_train(20_000)

    print("\nSelf-play della rete neurale (NFSP):")
    agents, xs, ys = run()

    learned = current_average_strategy(agents)

    # ---- grafico di convergenza ----
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, lw=2, color="#2563eb")
    plt.axhline(0, color="gray", lw=0.8, ls="--")
    plt.xlabel("Partite di self-play")
    plt.ylabel("Exploitability  (0 = ottimo / Nash)")
    plt.title("La rete neurale impara da sola a giocare a poker (NFSP)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("convergenza.png", dpi=130)
    print("\nGrafico salvato: convergenza.png")

    # ---- tabella di confronto rete vs Nash esatto ----
    print("\nConfronto strategia IMPARATA dalla rete  vs  ottimo teorico (CFR)")
    print(f"{'situazione':<26}{'rete P(bet)':>13}{'Nash P(bet)':>13}")
    for c in range(3):
        for h in kuhn.HISTORIES:
            p = kuhn.player_to_act(h)
            if (p == 0 and h in kuhn.P0_HISTORIES) or (p == 1 and h in kuhn.P1_HISTORIES):
                key = kuhn.infoset_key(c, h)
                lbl = f"P{p} carta {CARD[c]} storia {h or '(inizio)'}"
                print(f"{lbl:<26}{learned[key]:>13.3f}{nash[key]:>13.3f}")

    print(f"\nExploitability finale rete : {kuhn.exploitability(learned, learned):.4f}")
    print(f"Exploitability equilibrio  : {kuhn.exploitability(nash, nash):.4f}")
