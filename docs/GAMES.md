# The Games

This guide covers the seven games in the project. For each one it gives the rules as implemented, how to play it in the web app, how the rule-based **System** bot plays, and how the trained **agent** works.

The rules engines in `card_games_rl/games/<game>/rules.py` are the source of truth. The technical specification (observations, actions, rewards) is in [ENV_DESIGN.md](ENV_DESIGN.md).

| # | Game | Players | Type | Agent method |
|---|---|---|---|---|
| 1 | Blackjack | 1 vs dealer | Dealer game | Reinforcement learning (PPO) |
| 2 | Klondike Solitaire | 1 | Solo puzzle | Reinforcement learning (PPO) |
| 3 | Texas Hold'em | 2 | Competitive, hidden information | Counterfactual regret minimisation (CFR) |
| 4 | Rummy | 2 | Competitive, hidden information | Look-ahead search |
| 5 | Bluff | 4 | Competitive, deception | Reinforcement learning (PPO, self-play) |
| 6 | Hearts | 4 | Competitive, trick-taking | Reinforcement learning (PPO, self-play) |
| 7 | Donkey | 4 | Competitive, shedding | Reinforcement learning (PPO, self-play) |

**Play modes**, available for every game:

- **Human vs System:** you against the rule-based bot (or the dealer).
- **Human vs Agent:** in the multi-player games, you against the trained agent. In Blackjack and Klondike it is a duel: you and the agent play the very same cards side by side.
- **Agent vs Agent:** watch. In the multi-player games, any seat can be switched to you, the System bot, the agent or a random player.

---

## 1. Blackjack

**Goal:** finish closer to 21 than the dealer without going over.

### Rules

- Cards 2–10 count their number, J/Q/K count 10, and an Ace counts 11 or 1. A hand with an Ace counted as 11 is *soft*.
- Each hand is dealt from a fresh six-deck shoe, so counting cards gives no edge.
- You and the dealer get two cards each; one dealer card stays face down until you finish.
- **Your actions:**

| Action | Meaning |
|---|---|
| Hit | Take another card |
| Stand | Keep your total |
| Double | First two cards only: double the bet, take one card, stop |
| Split | Turn a pair into two hands, once per hand; doubling after a split is allowed |
| Surrender | First two cards only: lose half the bet and quit the hand (late surrender) |
| Insurance | Offered when the dealer shows an Ace: a side bet of half the stake that pays 2:1 if the dealer has blackjack |

- The dealer draws to 17 and stands on soft 17.
- **Payouts:** win +1, blackjack +1.5 (3:2), push 0, loss −1, surrender −0.5.

### In the app

The legal actions appear as buttons. In Duel vs Agent, both of you play the same shoe, so the comparison is free of luck. The scoreboard tracks net units.

### System and agent

- **System bot:** plays like the dealer (hit below 17, never double, split or surrender). The real opponent is the dealer, which follows fixed rules.
- **Reference:** an exact dynamic-programming optimal strategy (`games/blackjack/optimal.py`).
- **Agent:** PPO with action masking, warm-started by imitation and then trained on millions of hands. Its observation is the hand total, softness, pair, dealer up-card and which actions are available.
- **Result:** −0.0006 ± 0.0096 units per hand on 50,000 held-out hands. That is within 0.01 of the optimal strategy on the same hands (+0.0013 there; over this sample both sit inside the noise around the true long-run value of perfect play, a small loss).

---

## 2. Klondike Solitaire

**Goal:** move all 52 cards to the four foundations, Ace to King by suit.

### Rules

- **Layout:**
  - seven tableau columns of 1–7 cards, with only the top card face up;
  - a 24-card stock;
  - a waste pile;
  - four foundations.
- Draw one card at a time from the stock to the waste. The waste can be recycled into the stock any number of times.
- Columns build down in alternating colours (a red 9 on a black 10). A face-up run can move as a unit, and only a King may fill an empty column.
- Exposed face-down cards turn over automatically, and safe moves to the foundations are made automatically. A foundation card may be moved back onto a column.
- **A deal ends when:**
  - it is won;
  - it is **deadlocked**: an exact search proves no sequence of moves can make progress;
  - it hits the move limit.
- Positions that repeat are avoided while any fresh move exists.

### In the app

- Click a card or the top of a run, then click its destination; valid targets are outlined.
- Clicking a picked-up card again sends it home.
- Click the stock to draw, or ↺ to recycle.
- **Undo** is available when you play alone.
- After a lost deal, **Was this deal winnable?** runs a perfect-information solver on the starting deal. About 77% of deals are provably winnable; some can't be won even with every card visible.

### System and agent

- **System bot:** priority rules: foundation moves, then revealing hidden cards, then playing the waste, then drawing. It wins about 51% of deals.
- **Agent:** PPO with action masking over 591 move types, trained on the GPU. Its observation includes every visible card's position, the stock and waste, and foundation progress. The rewards are the win and the number of cards on the foundations, plus potential-based shaping for revealing cards. The shaping cannot be farmed by moving cards back and forth.
- **Result:** 45% of validation deals won. The target was 60%, so this game's agent is still below target.

---

## 3. Texas Hold'em (heads-up no-limit)

**Goal:** win chips from your single opponent.

### Rules

- Each hand starts with 200 chips each (100 big blinds). The button posts the small blind (1) and the other player the big blind (2); the button alternates every hand.
- Each player gets two hole cards. Betting rounds happen:
  - preflop, where the button acts first;
  - on the flop (3 board cards), where the big blind acts first from here on;
  - on the turn (4th card);
  - on the river (5th card).
- **Actions:** fold, check/call, raise ½ pot, raise pot, raise 2× pot, all-in. Minimum-raise rules apply, and an incomplete all-in raise does not reopen the betting.
- At showdown the best five-card hand out of each player's seven cards wins; exact ties split the pot.
- **Hand ranks, low to high:** high card, pair, two pair, three of a kind, straight, flush, full house, four of a kind, straight flush.

### In the app

The D chip marks the button, and the pot and amount to call show in the middle. Bet buttons show the total you would raise to. The scoreboard keeps a running chip count.

### System and agent

- **System bot:** an equity and pot-odds player. It calls when its chance of winning beats the price, raises for value with strong hands, and bluffs occasionally.
- **Agent:** **Monte Carlo counterfactual regret minimisation**, the family of methods behind modern poker AIs. The game is abstracted for training:
  - hands are grouped into 10 strength buckets per street;
  - betting is limited to fold, call, ½ pot, pot and all-in, with at most 3 raises per street.
- **Training:** four worker processes ran external-sampling MCCFR (regret matching+, linear averaging), and their average strategies are merged. The agent plays a *mixed* strategy, so it randomises, bluffs and slow-plays. In the rare situations training never reached, it uses the System bot's rule.
- **Result:** +25.9 bb/100 against the System bot on validation hands (target +10). Every deal was played from both seats to reduce luck.

---

## 4. Rummy (two players, 10 cards)

**Goal:** be the first to arrange your whole hand into melds.

### Rules

- **Melds:**
  - a *set* is 3–4 cards of one rank;
  - a *run* is 3+ consecutive cards of one suit. The Ace is low only: A-2-3 counts, Q-K-A does not.
- **A turn:** draw the top of the stock or the face-up discard, then discard one card. You can't throw back a card you just took from the discard pile.
- **Going out:** you win when, after your discard, every card left in your hand is in a meld.
- **Deadwood** is the value of unmelded cards: Ace 1, number cards their value, J/Q/K 10.
- When the stock runs out it is rebuilt from the discard pile once. If it runs out again, the lower deadwood wins.

### In the app

Click the stock or the discard to draw, then click a card to discard. Going out happens automatically. Your deadwood shows next to your name.

### System and agent

- **System bot:** greedy. It takes the discard only if that lowers its deadwood, and discards to minimise deadwood.
- **Agent:** a **look-ahead search**. For each possible discard it computes the expected deadwood after the next draw, averaged exactly over every unseen card, with a large bonus for draws that go out. It penalises discards that fit cards the opponent is known to hold. It uses only information the player legally has and needs no training. Monte Carlo rollouts were tried first and lost to the greedy bot, because the random outcomes drowned out the differences between moves.
- **Result:** 55.5% wins against the System bot on validation deals (target 55%). The bot against itself wins 50%.

---

## 5. Bluff (Indian rules, four players)

**Goal:** be first to empty your hand. Cards are played face down, so you may lie about them.

### Rules

1. **Start of a round:** the starter names any rank (Ace to King) and plays 1–4 cards face down, claiming they are that rank. The starter need not hold it.
2. **Later turns:** each following player either adds 1–4 cards claiming the same rank, or passes.
3. **Calling:** the player right after a claim may instead call bluff on it. The claimed cards are revealed:

| Result | Who picks up the pile | Who starts the next round |
|---|---|---|
| The claim was a lie | the claimant | the caller |
| The claim was true | the caller | the claimant |

4. **Everyone passes:** if all the other players pass after the last claim, the pile is discarded for good, and the player after the previous round's starter begins the next round.
5. **Winning:** play your last card and survive the next player's decision: they don't call, or they call and the claim was true.

### In the app

- **To start a round:** pick a rank, select 1–4 cards, and press *Claim*.
- **Later in the round:** select cards and press *Add*, or use *Pass* or *Call bluff*.
- You never learn whether a computer player bluffed unless someone calls. The agent's alternative moves are hidden while you play, because they could reveal its cards.

### System and agent

- **System bot:** honest by default and bluffs a single card now and then, less often as its hand shrinks. It always calls an impossible claim or a claimant's last card. Otherwise it calls more when the round's claims look inflated and less when the pile is large.
- **Agent:** PPO with self-play against a pool of its past versions plus System bots. Its observation includes:
  - its hand;
  - every player's card count;
  - the pile size;
  - the current rank and how many were claimed;
  - "certain lie" indicators from its own cards;
  - each opponent's call and caught-lying history.
- The reward is winning, plus shaping for shedding cards. Bluffing is not rewarded in itself, so it is learned only where it actually sheds cards.
- **Result:** 91.9% wins against three System bots on validation games (target 30%; a fair share is 25%).

---

## 6. Hearts (four players)

**Goal:** take as few points as possible. The match ends when someone reaches 100, and the lowest total wins.

### Rules

- Each hand starts with everyone passing three cards. The direction rotates: left, right, across, then no pass.
- The 2♣ leads the first trick. Players must follow suit if they can, and the highest card of the led suit wins the trick and leads next.
- **Points:** each heart is 1, and the Q♠ is 13. No points may be played on the first trick unless a player has nothing else. Hearts can't be led until one has been played.
- **Shooting the moon:** taking all 26 points gives every other player 26 instead.

### In the app

Click three cards to pass, then click a legal card to play; illegal cards are dimmed. Seats show their points this hand and in the match.

### System and agent

- **System bot:** passes dangerous high cards, ducks under the winning card when it can, and dumps the Q♠ and high hearts when void in the led suit.
- **Agent:** PPO with self-play. Its observation covers:
  - its hand;
  - all cards played so far;
  - the current trick;
  - the suits each player is known to be out of;
  - points taken.
- **Result:** 5.23 ± 0.31 points per hand against three System bots on 2,000 held-out test hands (target ≤ 6.0; the System bot against itself takes 7.2).

---

## 7. Donkey (Gadha Ladam / Kazhutha, four players)

**Goal:** get rid of all your cards. The last player still holding cards is the **Donkey** and loses.

### Rules

1. Four players get 13 cards each, and the holder of the A♠ leads it.
2. Going clockwise, each player must play the led suit if they can.
3. **Everyone follows:** the trick is discarded (out of the game) and the player of the highest card leads next, with any card.
4. **Someone can't follow:** they **cut** (punish) with any card. The trick stops immediately, and the player who played the highest card of the led suit picks up every card in the trick and leads next. Cutting is allowed even on the first trick.
5. A player whose hand becomes empty is out and safe. When only one player still holds cards, that player is the Donkey. If everyone left runs out on the same trick, there is no Donkey.
6. **Stalemate rule:** if the same position repeats three times, or after 300 tricks, the game stops. The player with the most cards is the Donkey, and players tied for the most share the loss.
7. **Scoring:** the Donkey scores −1 and everyone else +1/3. The agent also earns shaping for every card it sheds.

### In the app

- Click a legal card.
- A finished trick stays on the table until the next lead, with a message: "centre cleared", or "A punished B: B picks up N cards".
- The cut card has a red ring, and the card that picks up has a gold ring.
- Seats show the suits each player is known to be out of.

### System and agent

- **System bot:**
  - It leads low in suits nobody is known to be out of, so the trick is likely discarded.
  - When following, it stays under the top card if a cut is coming, and otherwise dumps its highest card of the suit.
  - It cuts with its highest card.
- **Agent:** PPO with self-play. Its observation includes:
  - its hand;
  - the cards discarded so far;
  - the current trick;
  - each player's card count, known void suits and known cards.
- **Result:** the Donkey in 15.8% of validation games against three System bots (target ≤ 18%; a fair share is 25%).

---

## How results are measured

- **Duplicate deals:** in multi-player games, every deal is replayed with the agent in each seat, which removes most of the luck of the cards. Results are given with a 95% confidence interval over deals.
- **Validation and test:** validation seeds are used during training and test seeds only for the final report. The web app shows the test result when `python -m card_games_rl.training.evaluate --all` has been run after training, and the validation score otherwise.
- **Rewards:** always the real game score plus potential-based shaping, which provably cannot change the best strategy or be farmed.
