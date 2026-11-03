// Short in-app guides: how each game is played and how to drive it here.
// Full rules: docs/GAMES.md. The engines are the source of truth.

export const GAME_META = {
  blackjack: { icon: "21", hue: 145, players: "Solo vs dealer" },
  klondike: { icon: "♠", hue: 200, players: "Solitaire" },
  poker: { icon: "♦", hue: 0, players: "2 players" },
  rummy: { icon: "♣", hue: 265, players: "2 players" },
  bluff: { icon: "?!", hue: 30, players: "4 players" },
  hearts: { icon: "♥", hue: 335, players: "4 players" },
  donkey: { icon: "D", hue: 50, players: "4 players" },
};

export const GUIDES = {
  blackjack: [
    "Finish closer to 21 than the dealer without going over. Aces count 1 or 11, faces 10.",
    "Hit, stand, double (one card, double stake), split a pair once, or surrender half the bet.",
    "Dealer stands on soft 17. A natural blackjack pays 3:2; insurance is offered against an ace.",
    "Duel vs Agent: you and the agent play the very same shoe side by side.",
  ],
  klondike: [
    "Move all 52 cards to the four foundations, Ace up to King by suit.",
    "Build columns down in alternating colours. Only a King may fill an empty column.",
    "Click a card (or a run), then click where it should go. Click it again to send it home.",
    "Draw one card at a time with unlimited recycles. Undo is available when playing alone.",
    "After a lost deal, ask the solver whether the deal was winnable at all.",
  ],
  poker: [
    "Heads-up no-limit hold'em: 200 chips each, blinds 1 and 2.",
    "Best five-card hand from your two hole cards and the five board cards wins at showdown.",
    "The button posts the small blind and acts first before the flop, last after it.",
    "Bet sizes: half pot, pot, twice the pot, or all-in. The button alternates each hand.",
  ],
  rummy: [
    "Each turn: draw from the stock or take the top discard, then discard one card.",
    "Melds are sets (3-4 of a rank) or runs (3+ in a suit, Ace low).",
    "Go out when all cards left after your discard form melds.",
    "If the stock runs out twice, the lower deadwood (unmelded points) wins.",
  ],
  bluff: [
    "Start a round by naming any rank and playing 1-4 cards face down. You may bluff.",
    "Everyone after adds cards claiming that same rank, or passes.",
    "Only the next player may call the last claim. A lie: the claimant takes the pile. True: the caller does.",
    "If all others pass, the pile is discarded for good. First to empty their hand wins.",
  ],
  hearts: [
    "Pass three cards (left, right, across, none), then the 2♣ leads the first trick.",
    "Follow suit if you can. The highest card of the led suit takes the trick.",
    "Each heart is 1 point and the Q♠ is 13. Points are bad; take all 26 to shoot the moon.",
    "Hearts can't be led until broken. The match ends at 100 points, lowest total wins.",
  ],
  donkey: [
    "The holder of the A♠ leads it. Follow the led suit if you can.",
    "If everyone follows, the trick is discarded and the highest card leads next.",
    "If you can't follow, cut with any card: the highest card of the led suit picks up the trick.",
    "Empty your hand to be safe. The last player still holding cards is the Donkey.",
  ],
};

export const SHORTCUTS = [
  ["Space", "Pause or resume computer moves"],
  ["S", "Step one computer move"],
  ["N", "Next deal (when the deal is finished)"],
  ["R", "Restart with the current settings"],
  ["H", "Open or close How to play"],
  ["?", "Show this help"],
];

// Full "How to play" per game, shown in the side panel as collapsible sections.
// Each section: { title, text?: string, items?: string[], table?: [[term, meaning], ...] }
export const RULES = {
  blackjack: [
    { title: "Goal", text: "Beat the dealer: finish closer to 21 than the dealer without going over (busting). You play alone against the dealer; there is no other player." },
    { title: "Card values", items: [
      "2–10 count their number. J, Q and K count 10.",
      "An Ace counts 11, or 1 if 11 would bust you. A hand with an Ace counted as 11 is soft (soft 17 = A + 6).",
      "Ace + a 10-value card as your first two cards is a blackjack (a natural).",
    ] },
    { title: "A hand, step by step", items: [
      "You and the dealer get two cards each. One dealer card is face down until you finish.",
      "If the dealer shows an Ace, you are offered insurance first.",
      "Take actions until you stand, bust, double or surrender.",
      "The dealer then reveals and draws until reaching 17 or more. The dealer stands on soft 17.",
      "Cards come from a fresh six-deck shoe every hand, so counting cards gives no edge here.",
    ] },
    { title: "Your actions", table: [
      ["Hit", "Take another card."],
      ["Stand", "Keep your total and end your turn."],
      ["Double", "First two cards only: double the bet, take exactly one card, then stop."],
      ["Split", "A pair becomes two hands with one bet each. Once per hand; doubling after a split is allowed."],
      ["Surrender", "First two cards only: give up and lose half the bet."],
      ["Insurance", "When the dealer shows an Ace: a side bet of half your stake that pays 2:1 if the dealer has blackjack."],
    ] },
    { title: "Payouts", table: [
      ["Win", "+1 unit"], ["Blackjack", "+1.5 units (3:2)"], ["Push (tie)", "0"],
      ["Lose or bust", "−1 unit"], ["Surrender", "−0.5 units"],
    ] },
    { title: "At this table", items: [
      "Your options appear as buttons; only legal ones are shown.",
      "Duel vs Agent: you and the agent play the very same shoe side by side, so results compare luck-free.",
      "The scoreboard tracks net units over the session. Perfect play still loses about 0.5% per hand: the house edge.",
    ] },
    { title: "Tips", items: [
      "Stand on hard 17 or more; hit hard 11 or less.",
      "Always split Aces and 8s; never split 10s or 5s.",
      "Insurance loses money in the long run.",
    ] },
  ],
  klondike: [
    { title: "Goal", text: "Move all 52 cards onto the four foundations, one per suit, each built up from Ace to King." },
    { title: "The layout", items: [
      "Tableau: 7 columns of 1 to 7 cards. Only the top card of each column starts face up.",
      "Stock: the remaining 24 cards, face down. Waste: where drawn cards land.",
      "Foundations: four empty piles at the top right, one per suit.",
    ] },
    { title: "Moves", items: [
      "Draw one card from the stock onto the waste. When the stock is empty, recycle the waste into a new stock, as often as you like.",
      "Move the top waste card onto a foundation or a column.",
      "On the columns, build down in alternating colours: a red 9 on a black 10.",
      "A run of face-up cards in sequence can move together.",
      "Only a King (or a run starting with a King) may fill an empty column.",
      "An exposed face-down card turns face up automatically.",
      "A foundation card may come back down onto a column when that helps.",
      "Cards that can safely go home are moved there automatically.",
    ] },
    { title: "How a deal ends", items: [
      "Won: all 52 cards are on the foundations.",
      "Deadlocked: the game proves that no sequence of moves can make progress any more.",
      "Move limit: very long deals stop after a fixed number of moves.",
      "Not every deal can be won, even with perfect play. After a lost deal, ask the solver whether this one was winnable.",
    ] },
    { title: "At this table", items: [
      "Click a card (or the top of a run) to pick it up; valid destinations are outlined. Click the destination to move.",
      "Click a picked-up card again to send it to its foundation.",
      "Click the stock to draw, or ↺ to recycle the waste.",
      "Undo takes back your last move when you play alone.",
      "“All legal moves” lists every move as a button if you prefer.",
    ] },
    { title: "Tips", items: [
      "Turning over face-down cards matters most; prefer moves that expose them.",
      "Don't empty a column unless you have a King ready for it.",
      "Avoid rushing low cards home if they're still needed to hold a column.",
    ] },
  ],
  poker: [
    { title: "Goal", text: "Win chips from your one opponent. Each hand you either make them fold or show down the better five-card hand." },
    { title: "Setup", items: [
      "Both players start each hand with 200 chips (100 big blinds).",
      "The button posts the small blind (1); the other player posts the big blind (2). The button alternates every hand.",
      "Each player gets two private hole cards.",
    ] },
    { title: "Streets", items: [
      "Preflop: betting starts with the button (small blind).",
      "Flop: three shared board cards, then betting. From here the big blind acts first.",
      "Turn: a fourth board card, then betting.",
      "River: the fifth board card, then a last round of betting.",
      "Showdown: if neither player folded, the best five-card hand from your two cards plus the five board cards wins the pot.",
    ] },
    { title: "Your actions", table: [
      ["Fold", "Give up the hand and the chips you put in."],
      ["Check", "Pass when there is nothing to call."],
      ["Call", "Match the opponent's bet."],
      ["Raise ½ pot / pot / 2× pot", "Bet more; the button shows the total you raise to."],
      ["All-in", "Bet every chip you have."],
    ] },
    { title: "Hand ranks (low to high)", items: [
      "High card → Pair → Two pair → Three of a kind → Straight → Flush → Full house → Four of a kind → Straight flush.",
      "Ties between equal ranks are broken by the highest cards; an exact tie splits the pot.",
    ] },
    { title: "At this table", items: [
      "The D chip marks the button. The pot and the amount to call show in the middle.",
      "The scoreboard keeps your running chip total across hands.",
      "The agent plays a mixed strategy (it randomises), so it can bluff and slow-play.",
    ] },
    { title: "Tips", items: [
      "Play more hands on the button: you act last after the flop.",
      "Compare the price of a call with the pot: a 10-chip call into a 30-chip pot needs about a 25% chance to win.",
    ] },
  ],
  rummy: [
    { title: "Goal", text: "Be first to arrange all your cards into melds and go out." },
    { title: "Melds", items: [
      "Set: 3 or 4 cards of the same rank (7♠ 7♥ 7♦).",
      "Run: 3 or more consecutive cards of one suit (4♥ 5♥ 6♥). The Ace is low only: A-2-3 counts, Q-K-A does not.",
      "Each card can belong to only one meld.",
    ] },
    { title: "A turn", items: [
      "Draw one card: the top of the face-down stock, or the face-up top of the discard pile.",
      "Then discard one card. You can't throw back the card you just took from the discard pile.",
    ] },
    { title: "Going out and scoring", items: [
      "You go out (and win) when, after your discard, all 10 cards in your hand form melds. It happens automatically.",
      "Deadwood is the value of cards not in a meld: Ace 1, number cards their value, J/Q/K 10.",
      "If the stock runs out, the discard pile is reshuffled into a new stock once. If it runs out again, the lower deadwood wins.",
    ] },
    { title: "At this table", items: [
      "Click the stock or the discard to draw, then click a card in your hand to discard it.",
      "Your current deadwood shows next to your name.",
      "Cards your opponent took from the discard pile are known to you; the agent tracks them too.",
    ] },
    { title: "Tips", items: [
      "Taking the discard tells your opponent what you collect; draw from the stock when the discard doesn't help.",
      "Throw high unmatched cards early: they cost the most deadwood.",
      "Avoid discarding cards that fit what the opponent picked up.",
    ] },
  ],
  bluff: [
    { title: "Goal", text: "Be the first of four players to get rid of all your cards. Cards are played face down, so you may lie about them." },
    { title: "Starting a round", items: [
      "The starter names any rank, Ace to King, and plays 1 to 4 cards face down, claiming they are all that rank.",
      "The starter need not hold that rank: the opening claim can already be a bluff.",
    ] },
    { title: "Later turns in the round", items: [
      "Each following player, in turn, either adds 1 to 4 cards claiming the same rank, or passes.",
      "The player right after a claim may instead call bluff on it. Nobody else can call that claim.",
    ] },
    { title: "Calling bluff", table: [
      ["The claim was a lie", "The claimant picks up the whole pile; the caller starts the next round."],
      ["The claim was true", "The caller picks up the whole pile; the claimant starts the next round."],
    ] },
    { title: "When everyone passes", items: [
      "If all the other players pass after the last claim, the pile is discarded for good.",
      "The player after the previous round's starter begins the next round.",
    ] },
    { title: "Winning", items: [
      "Play your last card and survive: the next player doesn't call, or calls and your claim was true.",
      "Very long games stop at a move limit without a winner.",
    ] },
    { title: "At this table", items: [
      "To start a round: pick a rank, select 1-4 cards, then press Claim.",
      "Later: select cards and press Add, or use Pass / Call bluff.",
      "“Quick plays” offers ready-made choices of true and bluff cards.",
      "You never see whether the computer bluffed unless someone calls.",
    ] },
    { title: "Tips", items: [
      "Count: if the claims this round plus the cards of that rank in your own hand exceed 4, someone is lying.",
      "Bluffing gets riskier the bigger the pile and the fewer cards you hold.",
    ] },
  ],
  hearts: [
    { title: "Goal", text: "Take as few points as possible. The match ends when someone reaches 100; the lowest total wins." },
    { title: "Passing", items: [
      "Each hand starts with everyone passing 3 cards.",
      "The direction rotates: left, right, across, then a hand without passing.",
    ] },
    { title: "Playing tricks", items: [
      "The holder of the 2♣ leads it to the first trick.",
      "Everyone must follow the led suit if they can; otherwise they may play any card.",
      "The highest card of the led suit wins the trick and leads the next one.",
      "No points may be played on the first trick (unless you hold nothing else).",
      "Hearts can't be led until one has been played (hearts are broken), unless you hold only hearts.",
    ] },
    { title: "Scoring", table: [
      ["Each heart", "1 point"], ["Q♠", "13 points"],
      ["Shooting the moon", "Take all 26 points and everyone else gets 26 instead; you get 0."],
    ] },
    { title: "At this table", items: [
      "Click three cards to pass, then click a legal card to play. Illegal cards are dimmed.",
      "Each seat shows its points this hand and its match total.",
    ] },
    { title: "Tips", items: [
      "Pass high spades (A♠, K♠) unless you hold enough spades to protect them.",
      "Empty a suit early so you can dump the Q♠ and hearts on others.",
      "Watch for someone collecting every heart; take one to stop a moon shot.",
    ] },
  ],
  donkey: [
    { title: "Goal", text: "Get rid of all your cards. The last player still holding cards is the Donkey and loses." },
    { title: "First trick", items: [
      "Four players get 13 cards each.",
      "The holder of the A♠ leads it.",
    ] },
    { title: "Following and cutting", items: [
      "Going clockwise, each player must play a card of the led suit if they have one.",
      "Everyone followed: the centre is cleared (those cards are out of the game) and the player of the highest card leads next, with any card.",
      "Someone can't follow: they cut (punish) with any card. The trick stops at once, and the player who played the highest card of the led suit picks up every card in it, then leads next.",
      "Cutting is allowed even on the first trick.",
    ] },
    { title: "Going out and the end", items: [
      "A player whose hand becomes empty is out and safe.",
      "When only one player still holds cards, that player is the Donkey.",
      "If everyone left runs out on the same trick, there is no Donkey.",
      "If the same position repeats three times, or after 300 tricks, the game stops: the player with the most cards is the Donkey (a tie shares the loss).",
    ] },
    { title: "At this table", items: [
      "Click a legal card; the others are dimmed.",
      "A finished trick stays on the table until the next lead. A cut card has a red ring; the card that picks up has a gold ring.",
      "Seats show the suits a player is known to be out of, and cards everyone saw them pick up.",
    ] },
    { title: "Tips", items: [
      "Get out of a suit early: being void lets you cut.",
      "Play high cards when you expect everyone to follow (the trick is cleared), and keep low ones when a cut is likely.",
    ] },
  ],
};
