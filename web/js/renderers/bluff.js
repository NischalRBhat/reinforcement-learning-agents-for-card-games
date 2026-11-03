// Bluff (Indian rules): start a round by naming any rank, then everyone adds cards
// claiming that rank, passes, or calls the previous claim.
import { cardEl, h } from "../cards.js";
import { actionBar, named, otherHand, position, POSITIONS_4, seatTag, viewerSeat } from "./common.js";

const CALL = 14;
const PASS = 15;
const selections = new Map(); // `${session}:${board}` -> { cards: Set, rank }

function defaultRank(hand) {
  const counts = new Map();
  for (const card of hand) counts.set(card.rank, (counts.get(card.rank) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const key = `${ctx.session.id}:${ctx.index}`;
  const live = ctx.board.human_turn;
  const canPlay = live && v.can_play_exact;
  const myHand = Array.isArray(v.hands[me]) ? v.hands[me] : [];
  const sel = selections.get(key) ?? { cards: new Set(), rank: null };
  for (const id of [...sel.cards]) if (!canPlay || !myHand.some((c) => c.id === id)) sel.cards.delete(id);
  if (v.phase !== "declare") sel.rank = null;
  else if (!sel.rank) sel.rank = defaultRank(myHand);
  selections.set(key, sel);
  const rankLabel = v.phase === "declare" ? sel.rank : v.round_rank;

  const toggle = (id) => {
    if (sel.cards.has(id)) sel.cards.delete(id);
    else if (sel.cards.size < 4) sel.cards.add(id);
    ctx.rerender();
  };

  const seats = [0, 1, 2, 3].filter((s) => s !== me).map((seat) => {
    const claimed = v.claims.filter((c) => c.player === seat).reduce((n, c) => n + c.count, 0);
    return h("div", { class: `seat ${POSITIONS_4[position(ctx.board, seat)]}` },
      seatTag(ctx, seat, claimed ? ` · claimed ${claimed}` : "", seat === v.round_starter && !v.done ? " · started" : ""),
      otherHand(v.hands[seat]));
  });

  const reveal = v.reveal;
  const center = h("div", { class: "seat center" },
    h("div", { class: "claim" }, v.done ? "" : v.round_rank ? `Round of ${v.round_rank}s` : "New round"),
    h("div", { class: "pile" }, v.pile_count ? cardEl(null) : h("div", { class: "slot" }),
      h("span", { class: "pile-count" }, `Pile ${v.pile_count} · discarded ${v.discarded}`)),
    v.last_claim ? h("div", { class: "note" }, `${ctx.seatName(v.last_claim.claimant)} claimed ${v.last_claim.count} × ${v.round_rank}`) : null,
    reveal ? h("div", { class: `reveal ${reveal.truthful ? "truth" : "lie"}` },
      `${ctx.seatName(reveal.caller)} called ${ctx.seatName(reveal.claimant)}: `,
      h("strong", {}, reveal.truthful ? "true" : "a lie"),
      h("div", { class: "hand" }, reveal.cards.map((c) => cardEl(c, { small: true }))),
      `${ctx.seatName(reveal.loser)} takes ${reveal.pile} card${reveal.pile === 1 ? "" : "s"}`) : null,
    !reveal && v.last_event ? h("div", { class: "note" }, named(ctx, v.last_event)) : null);

  const hand = myHand.map((card) => cardEl(card, {
    onClick: canPlay ? () => toggle(card.id) : null,
    selected: sel.cards.has(card.id),
  }));

  let controls = null;
  if (live) {
    const rankPicker = v.phase === "declare"
      ? h("div", { class: "rank-picker" }, "Claim rank:", v.ranks.map((r) =>
        h("button", {
          class: `rank ${sel.rank === r.label ? "selected" : ""}`, dataset: { rank: r.rank },
          onClick: () => { sel.rank = r.label; ctx.rerender(); },
        }, r.label)))
      : null;
    const n = sel.cards.size;
    const verb = v.phase === "respond" ? "Add" : "Claim";
    const play = canPlay ? h("button", {
      id: `play-cards-${ctx.index}`, class: "act primary", disabled: n === 0,
      onClick: () => {
        const cards = [...sel.cards];
        sel.cards.clear();
        const rank = v.ranks.find((r) => r.label === sel.rank)?.rank;
        if (v.phase === "declare") ctx.actClaim(cards, rank);
        else ctx.actCards(cards);
      },
    }, n ? `${verb} ${n} × ${rankLabel}` : `Select 1-4 cards to claim as ${rankLabel}`) : null;
    const respond = v.legal_actions.filter((a) => a.action === CALL || a.action === PASS);
    const prompt = v.phase === "declare" ? "Start a round: pick any rank (you may bluff it) and 1-4 cards."
      : v.phase === "lead" ? `Play 1-4 cards as ${v.round_rank}s.`
      : v.can_call ? `${ctx.seatName(v.last_claim.claimant)} claims ${v.last_claim.count} × ${v.round_rank}: call, pass, or add your own ${v.round_rank}s.`
      : `Add cards as ${v.round_rank}s, or pass.`;
    controls = [
      h("div", { class: "prompt" }, prompt),
      rankPicker,
      h("div", { class: "actions" }, play, respond.map((a) =>
        h("button", { class: `act ${a.action === CALL ? "danger" : ""}`, dataset: { action: a.action }, onClick: () => ctx.act(a.action) }, a.label))),
      v.phase !== "declare" && canPlay
        ? h("details", { class: "legal-moves" }, h("summary", {}, "Quick plays (auto-picked cards)"),
          actionBar(ctx, v.legal_actions.filter((a) => a.action < CALL)))
        : null,
    ];
  }

  root.append(h("div", { class: "felt table4 bluff" },
    ...seats,
    center,
    h("div", { class: "seat south" },
      seatTag(ctx, me),
      Array.isArray(v.hands[me]) ? h("div", { class: "hand wrap fan" }, hand) : otherHand(v.hands[me]),
      controls,
      v.done ? h("div", { class: "result" }, v.winner != null ? `${ctx.seatName(v.winner)} emptied their hand` : "Stopped at the move limit") : null)));
}
