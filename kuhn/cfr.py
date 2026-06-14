"""
cfr.py
======
CFR = Counterfactual Regret Minimization (Zinkevich et al., 2007).

E' l'algoritmo "classico" e provatamente convergente per i giochi a informazione
imperfetta a somma zero. NON usa reti neurali: tiene una tabella di "regret" per
ogni information set e, iterando, la STRATEGIA MEDIA converge all'equilibrio di Nash.

Lo usiamo come ORACOLO: ci da' la strategia ottimale esatta del Kuhn poker, con cui
poi misuriamo quanto bene la rete neurale (nfsp.py) ha davvero imparato.
"""

import numpy as np
import kuhn

ACTIONS = ["p", "b"]   # 0 = pass, 1 = bet


class Node:
    """Un information set: accumula regret e strategia media."""
    def __init__(self):
        self.regret_sum = np.zeros(2)
        self.strategy_sum = np.zeros(2)

    def strategy(self, reach):
        # Regret matching: gioco le azioni in proporzione al regret positivo.
        r = np.maximum(self.regret_sum, 0.0)
        s = r / r.sum() if r.sum() > 0 else np.array([0.5, 0.5])
        self.strategy_sum += reach * s          # pesata per la "reach probability"
        return s

    def average_strategy(self):
        total = self.strategy_sum.sum()
        return self.strategy_sum / total if total > 0 else np.array([0.5, 0.5])


nodes = {}


def cfr(history, cards, reach0, reach1):
    """Ricorsione CFR. Ritorna il valore atteso (per il giocatore di turno)."""
    if kuhn.is_terminal(history):
        # payoff sempre espresso per P0; lo giro per il giocatore di turno
        p = kuhn.player_to_act(history)
        u0 = kuhn.payoff_p0(cards, history)
        return u0 if p == 0 else -u0

    player = kuhn.player_to_act(history)
    key = kuhn.infoset_key(cards[player], history)
    node = nodes.setdefault(key, Node())

    reach = reach0 if player == 0 else reach1
    strat = node.strategy(reach)

    util = np.zeros(2)
    node_util = 0.0
    for a, action in enumerate(ACTIONS):
        if player == 0:
            v = -cfr(history + action, cards, reach0 * strat[a], reach1)
        else:
            v = -cfr(history + action, cards, reach0, reach1 * strat[a])
        util[a] = v
        node_util += strat[a] * v

    # Aggiorno i regret (pesati per la reach dell'AVVERSARIO: counterfactual)
    cf_reach = reach1 if player == 0 else reach0
    node.regret_sum += cf_reach * (util - node_util)
    return node_util


def train(iterations=20000):
    for _ in range(iterations):
        for cards in kuhn.DEALS:
            cfr("", cards, 1.0, 1.0)
    # Estraggo la strategia media (= Nash) come prob. di puntare 'b'
    nash = {}
    for key, node in nodes.items():
        nash[key] = node.average_strategy()[1]   # P(bet)
    return nash


if __name__ == "__main__":
    nash = train(20000)
    # Riempio eventuali infoset mai visitati (non capita in Kuhn, per sicurezza)
    for c in range(3):
        for h in kuhn.HISTORIES:
            nash.setdefault(kuhn.infoset_key(c, h), 0.5)
    expl = kuhn.exploitability(nash, nash)
    print(f"Exploitability dell'equilibrio CFR: {expl:.5f}   (deve essere ~0)")
    print("\nStrategia ottimale  P(bet)  per ogni situazione:")
    names = {0: "J", 1: "Q", 2: "K"}
    for c in range(3):
        for h in kuhn.HISTORIES:
            if kuhn.player_to_act(h) == 0 and h in kuhn.P0_HISTORIES or \
               kuhn.player_to_act(h) == 1 and h in kuhn.P1_HISTORIES:
                key = kuhn.infoset_key(c, h)
                pl = kuhn.player_to_act(h)
                hist_lbl = h if h else "(inizio)"
                print(f"  P{pl}  carta {names[c]:>1}  storia {hist_lbl:<7} -> bet {nash[key]:.3f}")
