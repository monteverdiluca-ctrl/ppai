"""
holdem.py
=========
Motore di TEXAS HOLD'EM (heads-up, No-Limit) con i veri giri di puntata e le
puntate espresse in BIG BLIND (BB), piu' un bot basato sull'EQUITY (Monte Carlo).

Giri di puntata (street):
  PREFLOP -> FLOP (3 carte comuni) -> TURN (4a) -> RIVER (5a) -> showdown.
  (Nel poker vero i giri di puntata sono 4: preflop + i 3 postflop.)

Tutto in BB: blind piccolo 0.5 BB, blind grande 1 BB, stack in BB. Cosi' le puntate
si ragionano come fanno i giocatori ("apro a 2.5 BB", "punto pot", ...).

La forza della mano si misura con l'EQUITY: la probabilita' di vincere stimata
simulando tante mani avversarie casuali (Monte Carlo) con la libreria `treys`.
"""

import random
from treys import Card, Evaluator, Deck

EVAL = Evaluator()


def equity(hole, board, n_sims=400):
    """Probabilita' che 'hole' (2 carte) vinca contro 1 avversario casuale,
    dato 'board' (0-5 carte comuni). Stima Monte Carlo."""
    known = hole + board
    wins = ties = 0
    for _ in range(n_sims):
        deck = Deck()
        deck.cards = [c for c in deck.cards if c not in known]
        random.shuffle(deck.cards)
        opp = deck.draw(2)
        need = 5 - len(board)
        full_board = board + (deck.draw(need) if need else [])
        my = EVAL.evaluate(full_board, hole)        # piu' basso = piu' forte
        ot = EVAL.evaluate(full_board, opp)
        if my < ot:   wins += 1
        elif my == ot: ties += 1
    return (wins + 0.5 * ties) / n_sims


# --------------------------------------------------------------------------
#  Stato del gioco passato agli agenti (cosa "vede" il bot per decidere)
# --------------------------------------------------------------------------
class View:
    def __init__(s, hole, board, street, pot, to_call, my_stack, opp_stack, bb):
        s.hole, s.board, s.street = hole, board, street
        s.pot, s.to_call = pot, to_call          # tutto in BB
        s.my_stack, s.opp_stack, s.bb = my_stack, opp_stack, bb


# --------------------------------------------------------------------------
#  Una mano heads-up No-Limit
# --------------------------------------------------------------------------
STREETS = ["preflop", "flop", "turn", "river"]


def play_hand(agent0, agent1, stack=100.0, verbose=False):
    """Gioca una mano. Ritorna il guadagno netto (in BB) del giocatore 0."""
    deck = Deck()
    holes = [deck.draw(2), deck.draw(2)]
    agents = [agent0, agent1]
    invested = [0.0, 0.0]                # quanto ha messo nel piatto ciascuno
    stacks = [stack, stack]
    folded = [False, False]

    # Blind: heads-up il bottone (giocatore 0) mette il piccolo (0.5), l'altro il grande (1)
    def put(p, amount):
        amount = min(amount, stacks[p])
        stacks[p] -= amount; invested[p] += amount
    put(0, 0.5); put(1, 1.0)

    board = []
    for street in STREETS:
        if street == "flop":  board += deck.draw(3)
        if street == "turn":  board += deck.draw(1)
        if street == "river": board += deck.draw(1)

        # ordine di azione: preflop inizia il bottone (0), postflop inizia l'altro (1)
        order = [0, 1] if street == "preflop" else [1, 0]
        last_raiser = None
        acted = {0: False, 1: False}
        while True:
            done = True
            for p in order:
                opp = 1 - p
                if folded[p] or folded[opp]:
                    done = True; break
                to_call = invested[opp] - invested[p]
                if acted[p] and to_call == 0:
                    continue                       # giro chiuso per lui
                done = False
                view = View(holes[p], board, street, sum(invested), to_call,
                            stacks[p], stacks[opp], 1.0)
                act, amt = agents[p](view)
                acted[p] = True
                if act == "fold" and to_call > 0:
                    folded[p] = True
                    if verbose: print(f"  {street}: P{p} fold")
                    break
                elif act in ("call", "check"):
                    put(p, to_call)
                    if verbose and to_call > 0: print(f"  {street}: P{p} call {to_call:.1f}")
                else:  # bet / raise: 'amt' = importo TOTALE puntato in questo giro (BB)
                    target = max(amt, invested[opp] + 1.0)        # min-raise grezzo
                    target = min(target, stacks[p] + invested[p]) # cap all-in
                    put(p, target - invested[p])
                    last_raiser = p
                    if verbose: print(f"  {street}: P{p} raise a {invested[p]:.1f} BB")
            if done or folded[0] or folded[1]:
                break
            # giro finito se le puntate sono pari e entrambi hanno agito
            if invested[0] == invested[1] and acted[0] and acted[1]:
                break
        if folded[0] or folded[1]:
            break

    pot = sum(invested)
    if folded[0]:   net0 = -invested[0]
    elif folded[1]: net0 = +invested[1]
    else:                                          # showdown
        r0 = EVAL.evaluate(board, holes[0]); r1 = EVAL.evaluate(board, holes[1])
        if r0 < r1:   net0 = +invested[1]
        elif r0 > r1: net0 = -invested[0]
        else:         net0 = 0.0
    return net0


# --------------------------------------------------------------------------
#  Alcuni agenti
# --------------------------------------------------------------------------
def calling_station(view):
    """Chiama sempre (utile come avversario di riferimento)."""
    return ("call", 0) if view.to_call > 0 else ("check", 0)


def equity_bot(view):
    """Decide in base a EQUITY vs POT ODDS, e dimensiona la puntata in BB."""
    eq = equity(view.hole, view.board, n_sims=300)
    pot, call = view.pot, view.to_call
    if call > 0:
        pot_odds = call / (pot + call)             # equity minima per chiamare
        if eq < pot_odds - 0.03:
            return ("fold", 0)
        if eq > 0.70:                              # mano forte: rilancia ~pot
            return ("raise", view.pot + call + pot)
        return ("call", 0)
    else:
        if eq > 0.62:                              # punta value ~2/3 pot
            return ("bet", round(0.66 * pot, 1) or 1.0)
        if eq > 0.50 and random.random() < 0.4:    # a volte punta/bluffa
            return ("bet", round(0.5 * pot, 1) or 1.0)
        return ("check", 0)


def simulate(agent0, agent1, hands=2000, stack=100.0):
    """Gioca tante mani alternando il bottone e riporta il win-rate in BB/100."""
    total = 0.0
    for i in range(hands):
        if i % 2 == 0:
            total += play_hand(agent0, agent1, stack)
        else:
            total -= play_hand(agent1, agent0, stack)   # scambio posizioni
    return 100.0 * total / hands                          # BB / 100 mani


if __name__ == "__main__":
    random.seed(1)
    print("Esempio di una mano (verbose):")
    play_hand(equity_bot, calling_station, verbose=True)

    print("\nEquity-bot  vs  calling-station  (2000 mani):")
    bb100 = simulate(equity_bot, calling_station, hands=2000)
    print(f"  win-rate equity-bot = {bb100:+.1f} BB/100")
