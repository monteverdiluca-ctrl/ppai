"""
nfsp.py
=======
NFSP = Neural Fictitious Self-Play  (Heinrich & Silver, DeepMind, 2016).
E' IL metodo "rete neurale + reinforcement learning + self-play" che converge
all'equilibrio di Nash nei giochi a informazione imperfetta come il poker.

IDEA (importante per la ricerca):
nei giochi a informazione imperfetta non esiste una mossa "migliore" fissa: la
strategia ottimale e' MISTA (probabilistica). Per ottenerla NFSP fa convivere due
reti per ogni giocatore:

  1) RETE Q  (la parte di REINFORCEMENT LEARNING):
     impara la "best response", cioe' come sfruttare al meglio l'avversario
     ATTUALE. Da sola e' sfruttabile (e' deterministica e prevedibile).

  2) RETE Pi (la parte di apprendimento SUPERVISIONATO):
     impara la MEDIA storica di tutte le best response giocate.
     E' questa media che converge all'equilibrio di Nash.

Durante il self-play ogni giocatore, a ogni partita, sceglie a caso se comportarsi
da "best response" (rete Q, con probabilita' eta) o da "strategia media" (rete Pi).
Le mosse fatte in modalita' best-response alimentano la rete Pi: cosi' Pi impara
"in media cosa conviene fare". Nessun dato esterno: imparano SOLO giocando tra loro.
"""

import numpy as np
import kuhn

rng = np.random.default_rng(0)


def reseed(seed):
    """Reimposta il generatore casuale (per ripetere gli esperimenti con seed diversi)."""
    global rng
    rng = np.random.default_rng(seed)

ACTIONS = [0, 1]                                  # 0 = pass ('p'),  1 = bet ('b')
HIST_INDEX = {"": 0, "p": 1, "b": 2, "pb": 3}     # per la codifica della storia
N_IN = 3 + 4                                       # one-hot carta (3) + one-hot storia (4)


def features(card, history):
    """Trasforma cio' che il giocatore osserva in un vettore numerico per la rete."""
    x = np.zeros(N_IN, dtype=np.float32)
    x[card] = 1.0                                  # quale carta ho
    x[3 + HIST_INDEX[history]] = 1.0               # com'e' andata la mano finora
    return x


# ---------------------------------------------------------------------------
#  Una piccola rete neurale (MLP) scritta a mano, con ottimizzatore Adam.
#  Architettura:  input(7) -> hidden(ReLU) -> output(2).
# ---------------------------------------------------------------------------
class MLP:
    def __init__(self, n_in, n_hidden, n_out, lr=0.01):
        self.lr = lr
        # Inizializzazione "He" (adatta a ReLU)
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self.params = ["W1", "b1", "W2", "b2"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}  # Adam
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)                    # ReLU
        z2 = a1 @ self.W2 + self.b2                 # logits / Q-values
        return z2, (X, z1, a1)

    def _step(self, grads):                         # un passo di Adam
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            g = grads[p]
            self.m[p] = b1 * self.m[p] + (1 - b1) * g
            self.v[p] = b2 * self.v[p] + (1 - b2) * (g * g)
            mhat = self.m[p] / (1 - b1 ** self.t)
            vhat = self.v[p] / (1 - b2 ** self.t)
            setattr(self, p, getattr(self, p) - self.lr * mhat / (np.sqrt(vhat) + eps))

    def backward(self, cache, dz2):
        """Backprop a partire dal gradiente sull'output (dz2)."""
        X, z1, a1 = cache
        gW2 = a1.T @ dz2
        gb2 = dz2.sum(0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (z1 > 0)                         # derivata della ReLU
        gW1 = X.T @ dz1
        gb1 = dz1.sum(0)
        self._step({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2})

    # --- due tipi di addestramento -----------------------------------------
    def train_regression(self, X, action_idx, target):
        """Per la rete Q: avvicina Q(s, a_scelta) al ritorno osservato (MSE)."""
        q, cache = self.forward(X)
        dz2 = np.zeros_like(q)
        n = len(X)
        idx = np.arange(n)
        dz2[idx, action_idx] = 2.0 * (q[idx, action_idx] - target) / n
        self.backward(cache, dz2)

    def train_policy(self, X, action_idx):
        """Per la rete Pi: impara a riprodurre le azioni di best-response (cross-entropy)."""
        logits, cache = self.forward(X)
        logits = logits - logits.max(1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(1, keepdims=True)
        n = len(X)
        dz2 = probs.copy()
        dz2[np.arange(n), action_idx] -= 1.0        # gradiente softmax + cross-entropy
        dz2 /= n
        self.backward(cache, dz2)

    def policy_probs(self, x):
        """Distribuzione di probabilita' sulle azioni (la 'giocata percentuale')."""
        logits, _ = self.forward(x[None, :])
        logits = logits[0] - logits[0].max()
        p = np.exp(logits)
        return p / p.sum()

    def q_values(self, x):
        q, _ = self.forward(x[None, :])
        return q[0]


# ---------------------------------------------------------------------------
#  Buffer di memoria
# ---------------------------------------------------------------------------
class CircularBuffer:
    """Memoria RL: tiene le esperienze piu' recenti (la best response insegue
    l'avversario attuale, quindi i dati vecchi vanno scartati)."""
    def __init__(self, cap): self.cap, self.data = cap, []
    def add(self, item):
        if len(self.data) < self.cap: self.data.append(item)
        else: self.data[rng.integers(self.cap)] = item
    def sample(self, n):
        idx = rng.integers(len(self.data), size=min(n, len(self.data)))
        return [self.data[i] for i in idx]


class ReservoirBuffer:
    """Memoria SL: campione UNIFORME di TUTTE le mosse di best-response viste finora
    (reservoir sampling). Serve a stimare la strategia MEDIA su tutta la storia."""
    def __init__(self, cap): self.cap, self.data, self.seen = cap, [], 0
    def add(self, item):
        self.seen += 1
        if len(self.data) < self.cap: self.data.append(item)
        else:
            j = rng.integers(self.seen)
            if j < self.cap: self.data[j] = item
    def sample(self, n):
        idx = rng.integers(len(self.data), size=min(n, len(self.data)))
        return [self.data[i] for i in idx]


# ---------------------------------------------------------------------------
#  Agente NFSP per un giocatore
# ---------------------------------------------------------------------------
class NFSPAgent:
    def __init__(self, hidden=64, lr=0.01, eta=0.20, eps=0.10):
        self.q_net = MLP(N_IN, hidden, 2, lr)          # best response (RL)
        self.pi_net = MLP(N_IN, hidden, 2, lr)         # strategia media (SL)
        self.rl_mem = CircularBuffer(60_000)
        self.sl_mem = ReservoirBuffer(400_000)
        self.eta, self.eps = eta, eps                  # anticipatory prob, esplorazione

    def act(self, card, history, best_response_mode):
        x = features(card, history)
        if best_response_mode:
            # rete Q, epsilon-greedy (esplora)
            a = rng.integers(2) if rng.random() < self.eps else int(np.argmax(self.q_net.q_values(x)))
        else:
            # rete Pi (strategia media)
            p = self.pi_net.policy_probs(x)
            a = int(rng.random() >= p[0])              # campiona dalla distribuzione
        return a, x

    def learn(self, batch=128):
        rl = self.rl_mem.sample(batch)
        if rl:
            X = np.stack([b[0] for b in rl]); A = np.array([b[1] for b in rl])
            G = np.array([b[2] for b in rl], np.float32)
            self.q_net.train_regression(X, A, G)       # Q(s,a) -> ritorno (MC control)
        sl = self.sl_mem.sample(batch)
        if sl:
            X = np.stack([b[0] for b in sl]); A = np.array([b[1] for b in sl])
            self.pi_net.train_policy(X, A)             # Pi impara la media delle best response

    def average_policy_bet_prob(self, card, history):
        return float(self.pi_net.policy_probs(features(card, history))[1])
