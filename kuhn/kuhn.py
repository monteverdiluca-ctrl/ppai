"""
kuhn.py
=======
Motore di gioco del KUHN POKER + strumenti di valutazione teorica.

Perche' Kuhn poker?
- E' il "poker minimo": 3 carte (J=0, Q=1, K=2), 2 giocatori, 1 carta a testa.
- Ha solo 12 situazioni di decisione (information set): abbastanza piccolo da
  poter calcolare ESATTAMENTE la strategia ottimale (equilibrio di Nash) e usarla
  come "verita' di riferimento" contro cui misurare la rete neurale.
- Mantiene pero' la caratteristica fondamentale del poker vero: l'INFORMAZIONE
  IMPERFETTA (non vedi la carta dell'avversario). E' qui che il RL diventa difficile.

Regole:
- Ogni giocatore mette 1 chip nel piatto (ante). Piatto iniziale = 2.
- Il Giocatore 0 (P0) agisce per primo. Due azioni possibili ovunque:
      'p' = pass  (= check se non c'e' una puntata, = fold se c'e')
      'b' = bet   (= punta 1 chip, oppure call di una puntata)
- Sequenze terminali e payoff per P0:
      'pp'  check-check          -> showdown, vince la carta piu' alta, +/-1
      'pbp' check-bet-fold(P0)   -> P0 si arrende: -1
      'pbb' check-bet-call       -> showdown, +/-2
      'bp'  bet-fold(P1)         -> P0 vince: +1
      'bb'  bet-call             -> showdown, +/-2
"""

from itertools import permutations, product

# Le 6 distribuzioni possibili (carta di P0, carta di P1), ognuna con prob 1/6
DEALS = list(permutations([0, 1, 2], 2))

# Storie in cui ogni giocatore deve decidere (servono per enumerare le strategie)
HISTORIES = ["", "p", "b", "pb"]          # tutte le storie non terminali
P0_HISTORIES = ["", "pb"]                 # storie in cui tocca a P0
P1_HISTORIES = ["p", "b"]                 # storie in cui tocca a P1


def player_to_act(history):
    """Di chi e' il turno: dipende solo dalla lunghezza della storia."""
    return len(history) % 2


def is_terminal(history):
    return history in ("pp", "pbp", "pbb", "bp", "bb")


def payoff_p0(cards, history):
    """Payoff per P0 in una storia terminale. (P1 prende l'opposto: gioco a somma zero.)"""
    c0, c1 = cards
    p0_wins = c0 > c1                      # showdown: vince la carta piu' alta
    if history == "pp":                    # check-check, piatto = 2 -> +/-1
        return 1 if p0_wins else -1
    if history == "pbp":                   # P0 ha foldato dopo la puntata di P1
        return -1
    if history == "bp":                    # P1 ha foldato dopo la puntata di P0
        return 1
    if history in ("pbb", "bb"):           # showdown con piatto = 4 -> +/-2
        return 2 if p0_wins else -2
    raise ValueError(f"Storia non terminale: {history}")


def infoset_key(card, history):
    """Chiave dell'information set: cio' che il giocatore di turno OSSERVA
    (la propria carta + la storia delle puntate). NON contiene la carta avversaria."""
    return f"{card}{history}"


# ---------------------------------------------------------------------------
#  Valutazione esatta: valore atteso e best response (per l'EXPLOITABILITY)
# ---------------------------------------------------------------------------
# Una "strategia" qui e' un dizionario:  infoset_key -> probabilita' di puntare ('b').

def _node_value(history, cards, s0, s1):
    """Valore atteso per P0 in un nodo, date le due strategie. Ricorsione sull'albero."""
    if is_terminal(history):
        return payoff_p0(cards, history)
    player = player_to_act(history)
    card = cards[player]
    key = infoset_key(card, history)
    p_bet = (s0 if player == 0 else s1)[key]        # prob. di giocare 'b'
    v_bet = _node_value(history + "b", cards, s0, s1)
    v_pass = _node_value(history + "p", cards, s0, s1)
    return p_bet * v_bet + (1 - p_bet) * v_pass


def expected_value(s0, s1):
    """Valore atteso del gioco per P0, mediando sulle 6 distribuzioni di carte."""
    return sum(_node_value("", cards, s0, s1) for cards in DEALS) / len(DEALS)


def _deterministic_strategies(histories):
    """Genera tutte le strategie deterministiche per le carte 0,1,2 sulle date storie."""
    keys = [infoset_key(c, h) for c in range(3) for h in histories]
    for bits in product([0, 1], repeat=len(keys)):
        yield {k: float(b) for k, b in zip(keys, bits)}


def best_response_value_p0(s1):
    """Miglior valore ottenibile da P0 se P1 gioca s1 (P0 sceglie la sua strategia)."""
    # Le strategie di P0 vivono sulle storie P0; per le storie P1 mettiamo placeholder.
    best = -1e9
    for s0_part in _deterministic_strategies(P0_HISTORIES):
        s0 = {infoset_key(c, h): 0.0 for c in range(3) for h in P1_HISTORIES}
        s0.update(s0_part)
        best = max(best, expected_value(s0, s1))
    return best


def best_response_value_p1(s0):
    """Miglior valore (per P1) se P0 gioca s0. P1 massimizza il PROPRIO payoff = -u0."""
    best = -1e9
    for s1_part in _deterministic_strategies(P1_HISTORIES):
        s1 = {infoset_key(c, h): 0.0 for c in range(3) for h in P0_HISTORIES}
        s1.update(s1_part)
        best = max(best, -expected_value(s0, s1))   # P1 vuole minimizzare u0
    return best


def exploitability(s0, s1):
    """
    Quanto e' lontana la coppia di strategie dall'equilibrio di Nash.
    = media di quanto ciascun giocatore guadagnerebbe deviando alla best response.
    All'equilibrio vale 0; piu' e' alta, piu' la strategia e' "sfruttabile".
    """
    br0 = best_response_value_p0(s1)     # quanto P0 puo' guadagnare vs s1
    br1 = best_response_value_p1(s0)     # quanto P1 puo' guadagnare vs s0
    return (br0 + br1) / 2.0


if __name__ == "__main__":
    # Strategia uniforme (50/50 ovunque): dovrebbe essere parecchio sfruttabile.
    uni = {infoset_key(c, h): 0.5 for c in range(3) for h in HISTORIES}
    print("Exploitability strategia uniforme:", round(exploitability(uni, uni), 4))
