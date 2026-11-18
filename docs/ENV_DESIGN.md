# Environment Design Spec

This is the reference specification for every game's rules, state, observation, actions, rewards and learning approach. The code in `card_games_rl/games/<game>/` implements exactly this. When a design decision changes, update this file first.

---

## 1. Shared design

### 1.1 Engine contract (`core/game.py`)

Every game is a pure-Python, seeded, deterministic engine with `N` seats:

| Member | Meaning |
|---|---|
| `reset(seed)` | Deal a new hand / episode |
| `current_player` | The seat that must act next (`-1` when terminal) |
| `legal_actions()` / `action_mask()` | The exact legal moves for `current_player` |
| `step(action)` | Apply a legal action; raises `IllegalAction` otherwise |
| `is_terminal`, `is_truncated` | Natural end / safety cap reached |
| `score(p)` | **Realized** utility of seat `p` so far; equals the final return at the end |
| `potential(p)` | Heuristic progress `Φ` used only for shaping (see 1.2) |
| `observation(p)` | Fixed-size `float32` vector: **only** information seat `p` can legally know |
| `view(viewer)` | JSON for the UI, with hidden cards masked for that viewer |

Cards are integers `0..51` (`suit*13 + rank-2`, rank 2..14 with Ace = 14, suits ♣♦♥♠). Card actions use **card ids**, never hand slots, so an action means the same thing whatever the hand size.

### 1.2 Reward framework

For the learner seat, the reward between two of its consecutive decisions (opponent moves in between are included) is:

```
r = [S(s') − S(s)]  +  λ · [Φ(s') − Φ(s)]        with Φ(terminal) = 0
```

- **S (realized score)** is the actual game result as it accrues, e.g. Hearts penalty points as tricks are taken, or chips at showdown. The sum of `S` deltas over an episode is exactly the game's final return. This is the true objective.
- **Φ (potential)** is optional shaping toward sub-goals (revealing Klondike cards, lowering Rummy deadwood). Because the shaping term is a difference of a potential, and Φ = 0 at terminal, its sum over any episode is the constant `−Φ(s₀)`. **It cannot change which policy is optimal, and it cannot be farmed**: cycling between states earns exactly 0 (Ng, Harada & Russell, 1999).
- Rewards accumulate on **every** state change, including opponents' turns. This fixes the old bug where a trick completed by an opponent lost the learner's penalty.
- There is no illegal-action penalty, because illegal actions are masked and cannot happen.

### 1.3 Seats, opponents and modes

The Gymnasium wrapper (`envs/single_agent.py`) gives one seat to the learner and drives the other seats with policies:

- `RandomPolicy`: uniform over legal moves.
- `HeuristicPolicy`: a rule-based "System" player for each game.
- Frozen self-play snapshots from the opponent pool.

The learner's seat is randomized each episode in multi-player games. The same seat abstraction powers the UI modes (human / system / agent per seat).

### 1.4 Learning approach overview

| Game | Information | Players | Recommended | Why not the alternatives (within a 2–4 h CPU budget) |
|---|---|---|---|---|
| Blackjack | Hidden dealer card, stochastic | 1 vs fixed dealer | **Exact DP optimal policy** (oracle) + **MaskablePPO** agent | The MDP is tiny and fully known, so DP is provably optimal; RL is trained to show it converges to it. DQN works too but adds nothing. |
| Klondike | Hidden face-down cards, long horizon | 1 | **MaskablePPO** (large MLP) + auto safe moves + stall truncation | DQN struggles with 591 actions and sparse wins; AlphaZero-style MCTS needs perfect information (Klondike's is hidden) and far more compute. |
| Poker (HU NLHE) | Imperfect info, zero-sum | 2 | **MaskablePPO self-play with an opponent pool** + all-in EV variance reduction | CFR / Deep CFR is the gold standard for low exploitability but needs card/bet abstraction and many CPU-hours; NFSP is several times slower to converge. PPO + pool beats heuristics in budget; Deep CFR is the documented upgrade path. |
| Rummy | Imperfect info | 2 | **MaskablePPO self-play** + deadwood potential | ISMCTS is strong but slow at inference in Python; DQN works but is less stable under self-play. |
| Bluff | Imperfect info, deception | 4 | **MaskablePPO self-play** with engineered memory features | CFR is infeasible with 4 players and the state size; recurrent PPO in sb3 has no action masking, so history is summarised in the observation instead. |
| Hearts | Imperfect info, trick-taking | 4 | **MaskablePPO self-play** (dense realized score) | ISMCTS is a strong but expensive inference-time upgrade; DQN is viable but PPO is more stable with 4-player non-stationarity. |

**MaskablePPO** (sb3-contrib) is PPO in which illegal actions get probability 0 in both training and inference.

Self-play opponent sampling, per episode and per opponent seat:

| Opponent source | Share |
|---|---|
| Latest 3 snapshots | 45% |
| Older snapshots | 25% |
| Heuristic (also the evaluation opponent) | 25% |
| Random | 5% |

Until the pool has snapshots, opponents are heuristic or random. The first snapshot is taken at 10% of the snapshot interval, so the pool fills quickly.

### 1.5 Training recipe (revised after the first full run)

1. **Behaviour-cloning warm start.** Before PPO, the policy head imitates a teacher and the value head regresses the teacher's returns. The teacher is the heuristic, or the exact optimal policy for Blackjack, playing against heuristic opponents. The cloned model is evaluated and saved as the first `best.zip`, so PPO can only improve on it.
   - **Why:** PPO from a random network spent 6M steps on Rummy and reached 0.7% against the heuristic. 26 seconds of cloning reached 28% (96% imitation accuracy).
2. **PPO with enough gradient steps.** `batch_size: 256` and `n_epochs: 10` give 40–80 gradient steps per rollout.
   - **Why:** the first configs used one minibatch per rollout, only 4 gradient steps, so networks learned about 50× more slowly per sample than intended.
3. **Self-play pool + heuristic share**, as in §1.4.
4. **Hardware layout:**
   - Env workers are torch-free: numpy snapshot inference, plus a custom vector env that returns action masks with every step.
   - Only the 6 trainer processes load PyTorch. `--device auto` puts each trainer on the GPU (CUDA) or CPU, whichever measured faster.

---

## 2. Blackjack

### Rules (variant)
- 6-deck shoe, shuffled every hand. Bet = 1 unit. Blackjack pays 3:2.
- The dealer takes a hole card and **stands on soft 17**.
- **Peek:** with an Ace up, insurance is offered first, then the dealer peeks. With a ten-value up card, the dealer peeks immediately. A dealer blackjack ends the hand: player blackjack pushes, otherwise the player loses 1 (insurance pays 2:1 on its 0.5 stake).
- **Double** on any first two cards, including after a split (DAS); exactly one more card, then the hand stands.
- **Split** a pair (same rank, or any two ten-value cards) once, giving two hands. Split Aces get one card each and auto-stand. 21 after a split is not a blackjack.
- **Late surrender** is allowed on the first two cards of the original hand only: lose 0.5.
- Bust > 21 loses immediately. The dealer plays only if some player hand is still live.

### State
True state: shoe order, dealer hole card, player hands, wagers, phase, active hand.
Observable state: everything except the shoe order and the hole card.

### Observation (31 floats, `OBS_VERSION = 1`)
| Slice | Feature |
|---|---|
| 0–2 | active hand total / 21, soft flag, card count / 8 |
| 3–12 | pair rank one-hot (2…10, A), all zero if the hand is not a splittable pair |
| 13–22 | dealer up card one-hot (2…10, A) |
| 23–29 | insurance phase, can double, can split, can surrender, is split hand, hand index, split aces |
| 30 | other split hand's total / 21 (0 if none) |

### Actions (7)
`0 hit, 1 stand, 2 double, 3 split, 4 surrender, 5 take insurance, 6 decline insurance`. In the insurance phase only 5 and 6 are legal.

### Rewards
| Event | Realized S |
|---|---|
| Any decision before resolution | 0 |
| Resolution | net units won, summed over hands + insurance, in [−4, +4] |

There is **no shaping (λ = 0)**: episodes are 1–6 decisions long and the true result arrives immediately, so shaping could only add bias. Hands that end at the deal (naturals, dealer peek blackjack) involve no decision. The training wrapper skips them; evaluation plays them, so the reported EV is the true EV.

### Baselines and gate
- Exact expectimax DP over (total, soft, pair, up card) with infinite-deck probabilities and dealer-peek conditioning. Expected EV for these rules ≈ −0.5%.
- **Gate:** the agent's EV/hand is within 0.01 of the DP policy's EV on the same 200k evaluation hands.

---

## 3. Klondike Solitaire

### Rules (variant)
- Standard deal: 7 columns of 1–7 cards, top card face up; 24 cards in the stock.
- **Draw-1**, unlimited recycles (waste turned back into the stock when the stock is empty).
- Foundations are built up by suit from the Ace. The tableau builds down in alternating colours; any face-up valid sequence can move. **Only Kings** (or King-led sequences) go into empty columns.
- A foundation card may move back onto the tableau (standard rule).
- An exposed face-down card **auto-flips**.
- **Auto safe moves:** a card goes to the foundation automatically when its rank ≤ 2, or when both opposite-colour foundations are ≥ rank − 1. This is provably never harmful, and it shortens the horizon. It is applied for humans too, as in most Klondike apps.
- **No-op move excluded:** moving a King-led sequence that already sits at the bottom of a column into another empty column.
- **No repeated positions (preference):** a move that recreates a position already seen since the last *progress* is hidden, unless every move would. Progress = a new best of foundation cards + revealed cards + stock cards seen. This removes back-and-forth loops. Blocking repeats outright was tried and rejected: the solver showed it sometimes cut the only path to a win.
- **Dead-end detection:** every 8 moves without progress, the engine searches every position reachable without progress (up to 4,000 positions). If none allows progress, the game ends as **deadlocked** (a loss). Every searched position has the same face-down cards, so this uses only information the player has. It was verified against the solver: no deadlocked position was ever winnable.
- **End:** win (all 52 cards on foundations), deadlocked, no legal moves, or the safety limits of 200 moves without progress / 1,000 moves.
- **Solver** (`klondike/solver.py`, analysis only): depth-first search with every card visible. The stock and waste are one ring, because with draw-1 and unlimited recycles any of them can be reached. It returns winnable / not winnable / unknown (budget exhausted). It is used for the evaluation report and for the UI's "Was this deal winnable?" after a game, never for choosing moves.
- **Undo:** humans only, in solo Play mode, handled by the API. It is not available to agents (it would leak face-down cards) or in the Duel.

### State
True state: tableau order including face-down cards, stock order, waste, foundations.
Observable state: face-up cards, waste, foundations, face-down counts per column, and stock cards **already seen** (in draw-1 every stock card is seen on the first pass).

### Observation (639 floats, `OBS_VERSION = 1`)
- Per card (52 × 12 one-hot location):
  - tableau column 0–6, face up (7)
  - waste top
  - waste below top
  - stock (already seen)
  - foundation
  - unknown (face-down, or stock not yet seen)
- Globals (15):
  - face-down count per column / 6 (7)
  - stock size / 24
  - waste size / 24
  - foundation height per suit / 13 (4)
  - stall counter / 200
  - action count / 1000

### Actions (591)
| Range | Move |
|---|---|
| 0 | draw stock → waste |
| 1 | recycle waste → stock |
| 2 | waste → foundation |
| 3–9 | waste → tableau column c |
| 10–16 | tableau column c → foundation |
| 17–44 | foundation suit s → tableau column c (`17 + s·7 + c`) |
| 45–590 | tableau src → tableau dst moving the top n face-up cards: `45 + (src·6 + dst′)·13 + (n−1)`, where dst′ skips src |

### Rewards
| Component | Value | Rationale |
|---|---|---|
| S: card reaches the foundation | +0.5/52 each (−0.5/52 if moved back) | Partial credit aligned with the goal |
| S: win | +0.5 | The true objective dominates: max return = 1.0 |
| Φ: revealed face-down cards | Φ = 0.25 · revealed / 21 (λ = 1) | Uncovering cards is the key sub-goal; potential-based, so shuffling cards back and forth earns 0 |
| Step cost | −0.001 per action | Discourages aimless stock cycling; it is negative, so it cannot be farmed |

The old design's +0.08 per tableau move was farmable and has been removed.

### Baselines and gate
- Random policy; priority-rule heuristic (foundation > reveal > build > draw).
- **Measured** with repetition avoidance and dead-end detection (150 deals): random play wins ~40%, the heuristic ~51% (it was ~35% before, because it looped).
- **Solver** on 60 test deals: 77% proven winnable, 3% proven not winnable, 20% unknown within 100k positions.
- **Gate:** win rate ≥ 60% on held-out deals. The report also gives the win rate on proven-winnable deals, the fairest measure.

---

## 4. Poker: heads-up No-Limit Texas Hold'em

### Rules (variant)
- 2 players, 200-chip stacks (100 big blinds), blinds 1/2. Stacks reset every hand; the button alternates.
- **Preflop** the button (small blind) acts first; **postflop** the big blind acts first.
- **Legal actions:**
  - fold, only when facing a bet (folding when checking is free is dominated and removed);
  - check/call;
  - raise to `opp_bet + f · (pot after calling)` with f ∈ {½, 1, 2}, raised to at least the minimum full raise (`opp_bet + max(last raise size, 2)`). Sizes at or above the stack are not offered, and identical raise-to sizes are deduplicated;
  - all-in.
- An all-in below a full raise does **not reopen** the betting for a player who has already acted.
- A street ends when both players have acted since the last full raise and the bets are equal, or a player is all-in and has been called. Remaining streets are dealt out when the players are all-in.
- **Showdown:** best 5 of 7 cards; split pots on ties.

### State
True state: both players' hole cards, deck order.
Observable state: own hole cards, board, pot, stacks, bets, betting history.

### Observation (146 floats, `OBS_VERSION = 1`)
| Block | Size |
|---|---|
| own hole cards (one-hot) | 52 |
| board (one-hot) | 52 |
| street one-hot | 4 |
| is button | 1 |
| pot / 400, my stack / 200, opponent stack / 200, to call / 200, my street bet / 200, opponent street bet / 200, min raise-to / 200 | 7 |
| betting history: per street × {me, opponent}: raises (clipped at 4) / 4 and has-acted flag | 16 |
| made-hand category one-hot (high card … straight flush) | 9 |
| preflop features: pair, suited, high rank / 14, low rank / 14, gap / 12 | 5 |

### Actions (6)
`0 fold, 1 check/call, 2 raise ½ pot, 3 raise pot, 4 raise 2× pot, 5 all-in`

### Rewards
| Component | Value |
|---|---|
| S: end of hand | chips won / 200, in [−1, 1] |
| **All-in EV (training only)** | When both players are all-in before the river, S uses `equity × pot − invested`. Equity is computed exactly on the turn/flop and with 400-sample Monte Carlo preflop. This is unbiased variance reduction, used in professional poker AI evaluation |
| Shaping | none (λ = 0); equity-based shaping would just add noise |

### Baselines and gate
- Random, calling station, heuristic. The heuristic plays preflop by hand-strength tiers and postflop by made-hand category plus pot odds.
- **Gate:** > +10 bb/100 against the heuristic over 20k hands (duplicate format: each deal played from both seats; 95% confidence interval).

### Agent: Monte Carlo CFR (`poker/cfr.py`, config `algorithm: cfr`)
PPO self-play (−9.7 ± 40.6 bb/100) learns to exploit whoever it plays rather than to be hard to exploit, so the agent is now a CFR equilibrium approximation:
- **Card abstraction:** 10 strength buckets per street, from equity against a random hand (the preflop table, then 64-sample Monte Carlo).
- **Betting abstraction:** street, this street's actions, the pot at the start of the street (7 buckets) and button position. Imperfect recall of earlier streets.
- **Actions in training:** fold, check/call, ½ pot, pot, all-in, with at most 3 raises per street.
- **Algorithm:** external-sampling MCCFR with regret matching+ and linearly weighted averaging.
  - Sampled deals fix the board, so leaves use real showdowns; this is 15× faster than all-in EV.
  - 4 independent worker processes, whose average strategies are merged (a mix of equilibrium strategies is still an equilibrium in two-player zero-sum games).
- **Speed and results:** about 85 iterations/s per worker, roughly 9k information sets. A 6-minute test run improved steadily: −324 → −170 bb/100 against the System bot over 95k iterations. A full run is ~40× longer.
- **Play:** situations training never reached fall back to the System bot; coverage is reported at each evaluation (100% in tests).

---

## 5. Rummy (basic 10-card, 2 players)

### Rules (variant)
- 2 players, 10 cards each; the rest forms the stock, with its top card turned up to start the discard pile. The non-dealer starts.
- **Turn:** draw one card (top of the stock, or top of the discard pile), then discard one card. A card just taken from the discard pile cannot be discarded on the same turn.
- **Melds:** sets of 3–4 of one rank; runs of 3+ consecutive cards of one suit, with the **Ace low only** (A-2-3 is valid, Q-K-A is not).
- **Going out:** if after the discard all 10 remaining cards can be partitioned into melds, the player goes out and wins. This is applied automatically, because going out is always optimal.
- **Stock exhausted:** the discard pile except its top card is turned over to form the new stock, once. On a second exhaustion, or after 100 turns each, the hand is a draw scored by deadwood.
- **Scoring:** the winner scores the loser's **deadwood**: the minimum total of unmelded cards over all meld arrangements, with A = 1, 2–10 at face value, J/Q/K = 10.

### State
True state: opponent's hand, stock order.
Observable state: own hand, the full discard history (each discard is public), and cards the opponent took from the discard pile (now known to be in their hand).

### Observation (320 floats, `OBS_VERSION = 2`)
- One-hot blocks, 52 each:
  - own hand
  - discard top
  - other public discards
  - known opponent cards
- Phase one-hot (draw / discard): 2
- **Meld-aware features (v2)**, all computed from the player's own hand:
  - per-card discard value (52): deadwood left after discarding that card, / 100 (1 where not applicable)
  - per-card meld membership (52): the card can be part of some meld
- Scalars (6):
  - own deadwood / 100
  - stock size / 31
  - turn / 100
  - opponent hand size / 11
  - melded-card fraction
  - take-discard gain: how much taking the top discard would lower the best achievable deadwood, / 100

**Why v2:** with v1 (raw one-hots only), 6M training steps produced an agent that never took the discard, made suboptimal discards 22% of the time, and won 0.7% against the heuristic. Reward and observation traces were verified correct. The bottleneck was learning combinatorial meld structure from one-hots with an MLP, so the features now expose meld arithmetic and the network learns strategy on top.

### Actions (54)
`0 draw stock, 1 draw discard, 2 + c discard card c` (c = card id)

### Rewards
| Component | Value |
|---|---|
| S: win | +(0.5 + 0.5 · min(opponent deadwood, 100) / 100); the loser gets the negative (zero-sum) |
| S: draw | (opponent deadwood − own deadwood) / 200 |
| Φ: deadwood potential | Φ = −0.5 · D / 100 (λ = 1), where D is the own deadwood (for 11 cards, the best deadwood after the best discard). Rewards forming melds immediately; potential-based, so it can't be farmed |

### Baselines and gate
- Random; greedy heuristic: take the discard if it lowers deadwood, otherwise draw from the stock, then discard the highest deadwood card.
- **Gate:** win rate > 55% against the heuristic.

### Agent: look-ahead search (`rummy/search.py`, config `algorithm: search`)
PPO reached 38.9% against the greedy bot. Monte Carlo rollouts with greedy continuations were tried next and also lost (42–49%): the noise of whole simulated games swamped the small differences between candidate moves. The agent is now an exact, deterministic one-draw look-ahead:
- **Discards:** each shortlisted discard is scored by the expected deadwood after the next stock draw, over every unseen card (going out counts as −60). A small penalty (3 per link, tuned on validation deals) applies when the card fits the opponent's known cards.
- **Draw:** take the discard only if the best resulting hand beats the expected value of drawing.
- **Result:** 54.3% ± 4% against the greedy bot on 600 validation deals (the bot against itself: 49.8%), 7 ms per decision, no weights. `train` validates and registers it.

---

## 6. Bluff (Indian rules), 4 players

### Rules (variant)
- 4 players, 13 cards each; a random player starts the first round.
- **Round start:** the starter names **any rank** (they need not hold it) and plays 1–4 cards face down.
- **Each later turn**, the player either:
  - **adds** 1–4 cards claiming the round's rank;
  - **passes**;
  - **calls** bluff on the previous claim, only right after it (the player directly after the claimant).
- **Call:** the claim's cards are revealed.
  - True: the caller takes the whole pile, and the claimant starts the next round.
  - A lie: the claimant takes the pile, and the caller starts.
- **Everyone else passes** after a claim: the pile is **discarded** (out of play), and the player after the previous round's starter starts the next round.
- **Win:** empty your hand and survive the next player's decision. If they pass or add, you win; if they call and the claim is true, you still win.
- **Truncation:** 1,000 actions; seats are then ranked by hand size (zero-sum). Measured with 4 System bots: median game 120 actions, none truncated.

### Observation (129 floats, `OBS_VERSION = 2`)
| Block | Size |
|---|---|
| own hand rank counts / 4 | 13 |
| phase one-hot (declare / lead / respond) | 3 |
| round rank one-hot | 13 |
| own count of the round rank / 4 | 1 |
| hand sizes by relative seat / 52 | 4 |
| pile, discarded, claims this round, cards claimed this round, cards claimed by others (clipped) | 5 |
| last claim size one-hot, claimant relative seat one-hot | 8 |
| can call, most cards of the rank others could hold / 4, certain lie, inflation of the round's claims, passes / 3 | 5 |
| own cards in the pile by rank, own discarded cards by rank | 26 |
| cards known to be in each opponent's hand, by rank (3 × 13) | 39 |
| per opponent: call rate, caught-lying rate (smoothed), claims made (3 × 3) | 9 |
| claimant is out of cards, actions taken / 1000, I started this round | 3 |

### Actions (29)
- `0–13` play (t true, b bluff) cards, 1 ≤ t + b ≤ 4. Bluff cards are chosen automatically, from ranks held in the fewest copies.
- `14` call
- `15` pass
- `16–28` name the round's rank (2…A) when starting a round. A two-step choice: the rank first, then the cards.
- Humans pick exact cards and a rank in one move (`step_cards(cards, rank)`).

Public labels hide how many cards were true. When a human is at the table, the UI shows only the agent's chosen move, not its alternatives.

### Rewards
| Component | Value |
|---|---|
| S: terminal | winner +1, others −1/3 (zero-sum). On truncation: hand-size rank, zero-sum |
| Φ: hand-size potential | Φ = 0.5 · (mean opponent hand − own hand) / 52 (λ = 1) |

Picking up a pile is felt immediately, and shedding cards (honestly or by bluffing) is rewarded. There is deliberately no bonus for bluffing itself, which could be farmed.

### Baselines and gate
- **System bot:**
  - starts rounds with its most-held rank and plays honestly;
  - bluffs one card sometimes, less often with a small hand;
  - calls certain lies and last-card claims, and otherwise calls when its calibrated lie estimate beats a threshold that rises with the pile size.
- **Measured:**
  - the bot wins 37.8% against 3 random players (fair share 25%);
  - a random player never wins against 3 bots;
  - a "dump cards, never call" player does not beat the bots (the old variant's exploit).
- **Gate:** win rate ≥ 30% against 3 System bots.

---

## 7. Hearts (Black Lady), 4 players

### Rules (variant)
- 4 players, 13 cards each.
- **Passing:** 3 cards, rotating left → right → across → none by hand number (randomized in training). Each player picks 3 cards; the cards are exchanged simultaneously.
- The holder of 2♣ leads it to the first trick. Players must follow suit if able; the highest card of the led suit wins and leads next.
- **First trick:** no hearts and no Q♠, unless the hand has nothing else.
- Hearts may not be **led** until broken (a heart has been played), unless the hand holds only hearts.
- **Points:** each heart = 1, Q♠ = 13. **Shooting the moon:** taking all 26 gives the shooter 0 and everyone else 26.
- Episode = one hand. The UI plays a match to 100 points.

### State
True state: all hands.
Observable state: own hand, cards played, current trick, passed and received cards, points taken by each seat, and voids revealed by failures to follow suit.

### Observation (446 floats, `OBS_VERSION = 1`)
| Block | Size |
|---|---|
| own hand (one-hot) | 52 |
| cards already played | 52 |
| current trick by relative seat | 4 × 52 |
| cards I passed | 52 |
| cards I received | 52 |
| pass direction one-hot | 4 |
| points taken by relative seat / 26 | 4 |
| known voids (3 opponents × 4 suits) | 12 |
| hearts broken, Q♠ played | 2 |
| phase (pass / play) | 2 |
| pass picks so far / 3 | 1 |
| trick number / 13 | 1 |
| leader relative seat one-hot | 4 |

### Actions (52)
Card id, for both passing and playing.

### Rewards
| Component | Value |
|---|---|
| S: after each trick | S_p = (mean points − points_p) / 26, updated as points are taken; a dense true signal, zero-sum |
| S: terminal | as above, with the shoot-the-moon adjustment |
| Shaping | none (λ = 0); the realized score is already dense |

### Baselines and gate
- Random; heuristic:
  - Pass: Q♠/K♠/A♠, high hearts and high cards from short suits.
  - Play: duck under the current winner when possible, dump Q♠ or high hearts when void, and lead low cards from short suits.
- **Gate:** average points per hand < 6.0 against 3 heuristics (6.5 = fair share).

---

## 7b. Donkey (Gadha Ladam), 4 players

### Rules
- 4 players, 13 cards each; the A♠ holder leads it to the first trick. Players must follow the led suit if they can.
- **Everyone still holding cards follows:** the trick is discarded (out of play). The highest card leads next; if that player is out, the next player clockwise with cards leads.
- **Someone can't follow:** they play any card (a **cut**, allowed on the first trick too). The trick stops at once, the player of the highest led-suit card **picks up** every card in it, and leads next.
- A player with no cards is safe. The last player holding cards is the **Donkey**. If everyone runs out together, there is no Donkey.
- **Stalemate:** two players can keep cards circulating forever. A position (all hands + leader) seen for the third time, or 300 tricks, stops the game: the player holding the most cards is the Donkey. A tie for most is shared.

### Observation (501 floats, `OBS_VERSION = 1`)
- Card blocks of 52: own hand; discarded cards; the current trick by relative seat (4 × 52); the cards known to be in each opponent's hand, because everyone saw them picked up (3 × 52).
- Known voids (3 opponents × 4 suits).
- Per relative seat: hand size / 52, still in the game.
- Led suit one-hot.
- Scalars: leading?, trick position / 3, first trick?, tricks played / 300, players left / 4.
- Seat holding the highest led-suit card, relative one-hot.

### Actions (52)
Card id, masked: the A♠ on the first lead, the led suit when the player has it, otherwise any card.

### Rewards
| Component | Value |
|---|---|
| S: terminal | Donkey −1, everyone else +1/3 (zero-sum; tied losers share −1) |
| Φ: card-count potential | Φ = 0.5 · (mean opponent cards − own cards) / 52 (λ = 1) |

Every card shed, whether discarded or passed to someone who picks up, is rewarded when it happens, and picking up is penalized at once. Being the Donkey decides the final outcome.

### Baselines and gate
- **System bot:** leads its lowest card in a suit no opponent is known to be void in; stays under the current highest card when a cut is coming, otherwise dumps its highest card; cuts with its highest card.
- **Measured:** the bot is the Donkey 3.2% of the time against 3 random players (fair share 25%), and a random player against 3 bots 59%.
- **Gate:** Donkey rate ≤ 18% against 3 System bots.

---

## 8. Test obligations (all games)

- **Rules scenario tests:** each bullet in the rules sections above has at least one targeted test.
- **Seeded fuzz** (random legal playouts, 300+ episodes per game), checking these invariants:
  - card conservation;
  - masks equal `legal_actions`;
  - termination within the cap;
  - identical seed gives an identical trajectory;
  - zero-sum returns where specified;
  - `observation(p)` and `view(p)` never encode hidden information. This is tested by altering hidden cards and asserting the observation is unchanged.
- **Gymnasium checks:** `check_env` passes; observations stay within their bounds; the sum of S-deltas equals the final return.

---

## 9. Card-counting agents (a second agent per game)

**Goal:** test whether exact card counting makes the agents stronger, without changing the current agents. Each counting agent is trained next to the current one and the two are compared on the same deals.

### Observation encoders (`card_games_rl/encoders.py`)

| Encoder | Contents | Version |
|---|---|---|
| `base` | the engine's `observation(p)`, exactly as before | `OBS_VERSION` |
| `counting` | `base` followed by `games/<g>/counting.features(game, p)` | (`OBS_VERSION`, `COUNTING_VERSION`) |

- **Base observations are frozen:** a test fingerprints the base observations on fixed deals, and they may not change without an `OBS_VERSION` bump.
- **No hidden information:** counting features use public information only. A test re-deals every card the player cannot locate among the other hands and asserts the features are unchanged.
- **Unseen cards:** cards the player cannot locate are treated as dealt at random into the opponents' unknown hand slots, respecting known voids. `core/counting.py` gives the closed forms (no match / at least *k* matches).

### Features

| Game | Size | Features |
|---|---|---|
| Donkey | +78 | **Per suit:** own, unseen and discarded counts; *lead safety* (every remaining opponent still holds the suit); expected opponents out of it; how low the player's lowest card sits among the unseen. **Per legal card:** the chance of ending up picking up the trick. It walks the players still to play: each may cut (the current highest card picks up) or follow and overtake (with probability ½ when it holds a higher card). **Scalars:** unseen cards; own cards the others know |
| Hearts | +88 | **Per suit:** own, unseen and played counts; chance a lead is cut; room to duck under the player's lowest card; risk above its highest. **Q♠:** where it is (hand, played, passed by the player, unseen) and each opponent's chance of holding it. **Points:** hearts unseen; points still out; leading opponent's points; moon-shot warning; points in the trick. **Per legal card:** the chance of winning the trick |
| Bluff | +20 | **Per rank:** unlocated copies (not in the player's hand, pile contributions or discards, nor seen going to someone after a call). The chance the last claim is true; claim inflation; *exposure* (the chance the next player holds enough copies to know a claim of 1–4 cards is a lie); unlocated total |
| Rummy | — | The counting agent is the look-ahead search plus an opponent model read from the public history. Discards that fit cards the opponent threw away or declined are safer. Unseen cards that fit the opponent's pick-ups are weighted down in the next-draw odds (they are probably in its hand); those that fit what it let go are weighted up |

### Training, storage and comparison

- **Training:** `train --variant counting` uses the same config, recipe, warm start and budget. The env, workers, self-play pool, policies and pretraining all receive the encoder. A `variants: {counting: {...}}` section in a game's YAML can override settings; Rummy uses it to name its counting search.
- **Storage:** models go to `models/<game>-counting/` with their own `meta.json` (`variant`, `obs_version`, `counting_version`), self-play pool and checkpoints. The registry refuses a counting model whose versions don't match.
- **Evaluation:** `evaluate --variant counting` reports against the same baselines and gate. It also runs a **head-to-head** against the base agent on duplicate deals: in 2-player games each deal is played with the agents in both seats; in 4-player games the counting agent takes two opposite seats and the base agent the others, then they swap. The report gives both agents' metric, the difference and its 95% CI.
- **App:** seats can be a *Counting agent*, and competitive games with a counting agent get an **Agent vs Counting agent** mode.


---

## 10. Canasta (4 players, two partnerships)

### Rules (variant)
Classic Hoyle partnership Canasta, as in docs/GAMES.md §8. Choices the rules leave open:
- **No permission to go out:** going out doesn't need the partner's permission.
- **Hand ends:** when the stock runs out, the hand ends as soon as the player to draw can't take the pile. If they can, the take is forced.
- **Pickup visibility:** pile pickups are public, so every player knows the kinds the taker picked up until they are melded or discarded.
- **Cap:** a hand stops at 800 actions, or in a rare dead end (for example one card left, no canasta and an empty stock), and is scored as if the stock had run out.

### Cards and kinds
Ids 0..107: `deck * 52 + card` for the two decks, 104..107 jokers. Melding and discarding use 15 kinds: joker, 2, red 3, black 3, then ranks 4..A.

### Actions (499)
| Range | Action |
|---|---|
| 0 | draw from the stock |
| 1–3 | take the pile: natural pair / natural + wild / add the top card to the side's meld |
| 4–483 | meld: target (ranks 4..A, black 3) × naturals added (0–3) × wild cards added (jokers j, twos d, j + d ≤ 3) |
| 484–498 | discard one card of a kind |

Masks enforce every rule: meld shape, wild limits, black threes only when going out, no emptying the hand without a canasta, and the opening minimum. An opening meld may be laid in several steps, but each step must keep the minimum reachable from the hand (a greedy upper bound), and discarding is blocked until it is reached.

### Agent actions: planner (21, `games/canasta/plan.py`, `PLAN_VERSION = 1`)
With the engine's actions a player makes about 150 decisions per hand, and the first agent (PPO on engine actions) lost 121 ± 42 points per hand to a System partnership: even its imitation of the System bot played far worse than the bot. The agent therefore chooses *planner actions*, and the System bot's meld rules fill in the details:

| Plan action | Engine action |
|---|---|
| 0–3 | draw / take the pile, unchanged |
| 4 `MELD` | the System bot's meld now (naturals; wild cards only to open, complete a canasta or go out) |
| 5 `MELD_WILD` | the meld with wild cards that grows a meld furthest (a canasta first, then size, 2s before jokers) |
| 6–20 | discard one card of a kind, unchanged |

- Choosing a discard while melds are possible is how the agent holds cards back.
- Every engine move stays reachable: any natural meld is a positive System meld and any meld with wild cards is a `MELD_WILD` candidate.
- The System bot's own moves translate exactly, so the warm start copies it perfectly (imitation accuracy 1.000, and the copy scores 0 against a System partnership).
- About 23 decisions per hand for the learner.
- The encoder (`card_games_rl/encoders.py`) owns the translation: policies return engine actions, so opponents, the API and evaluation are unchanged.
- **Planner features** (48, appended to the observation): the System bot's choice now (one-hot over the 21 plan actions); for `MELD` and `MELD_WILD`: available, cards added, wild cards used, new meld, makes a canasta, goes out; the System bot's discard choice (one-hot over 15 kinds).

### Observation (239 floats, `OBS_VERSION = 1`)
- Hand: count per kind.
- Melds of both sides per target: naturals, wilds, canasta, natural canasta.
- Red threes per side.
- Pile: size, top kind, frozen, blocked, and count per kind (every discard is seen).
- Hand sizes; stock size.
- Per side: melded or not, opening minimums, canasta counts, match scores.
- Turn: phase, value melded so far towards the opening, pile cards waiting for the opening.
- **Card counting:**
  - unseen copies per kind;
  - kinds each other player is known to hold from pickups;
  - per kind, the chance the next opponent could take the pile if that kind were discarded now.

### Rewards
| Component | Value |
|---|---|
| S: end of hand | (own side's hand score − other side's) / 2000 for both partners (zero-sum) |
| Φ: public progress | 0.5 · (progress(own side) − progress(other side)) / 2000, where progress counts melded card values, canasta bonuses, and red threes once the side has melded (λ = 1) |

Earlier versions also credited 30 points per card in melds still short of a canasta. That was removed: it pushed the agent to lay cards down early, which shows the opponents your ranks and spends the pairs needed to take the pile.

### Agent and evaluation
- **Training:** MaskablePPO over planner actions, warm-started as an exact copy of the System bot (policy and value head). Half of the partners and opponents are System bots (the evaluation opponent), the rest past versions; no random players. Small, careful PPO steps (learning rate 1e-4, clip 0.1, 4 epochs, entropy 0.003) refine the copied play instead of overwriting it. Network 256×256. Budget: 6 hours.
- **Evaluation:** the agent plays both seats of a side (`team_seats`) against a System partnership on duplicate deals.
- **Metric:** points per hand above the other side.
- **Gate:** ≥ +150 points per hand.
