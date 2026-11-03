// Donkey: follow suit to get cards discarded; cut when void so the highest card picks up.
import { cardEl, h, suitSymbol, SUIT_ORDER } from "../cards.js";
import { otherHand, position, POSITIONS_4, prompt, seatTag, viewerSeat } from "./common.js";

function voids(v, seat) {
  const list = v.voids[seat];
  return list.length ? ` · out of ${list.map((s) => suitSymbol(SUIT_ORDER[s])).join("")}` : "";
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const live = ctx.board.human_turn;
  const legal = new Set(v.legal_actions.map((a) => a.action));
  const status = (seat) => (v.out_order.includes(seat) ? " · safe" : voids(v, seat));

  const seats = [0, 1, 2, 3].filter((s) => s !== me).map((seat) =>
    h("div", { class: `seat ${POSITIONS_4[position(ctx.board, seat)]}` },
      seatTag(ctx, seat, status(seat)),
      otherHand(v.hands[seat]),
      v.known[seat].length && !Array.isArray(v.hands[seat])
        ? h("div", { class: "note" }, "Known: ", v.known[seat].map((c) => c.label).join(" ")) : null));

  // Until the next card is led, the finished trick stays on the table with what happened to it.
  const last = v.last_trick;
  const finished = !v.trick.length && last ? last : null;
  const plays = finished ? finished.plays : v.trick;
  const cutCard = finished?.type === "picked" ? plays[plays.length - 1].card.id : null;
  const pickerCard = finished?.type === "picked" ? plays.find((p) => p.player === finished.picker)?.card.id : null;
  const trick = h("div", { class: `trick ${finished ? `resolved ${finished.type}` : ""}` }, plays.map(({ player, card }) =>
    h("div", { class: `trick-card ${POSITIONS_4[position(ctx.board, player)]}` },
      cardEl(card, { className: card.id === cutCard ? "cut" : card.id === pickerCard ? "picker" : "" }))));

  const name = (seat) => ctx.seatName(seat);
  const verb = (seat, word) => (seat === ctx.board.human_seat ? word : `${word}s`);
  const outcome = !finished ? null
    : finished.type === "discarded"
      ? h("div", { class: "trick-outcome cleared" },
        h("strong", {}, "Centre cleared"),
        ` · everyone followed suit, ${finished.cards.length} cards discarded. ${name(finished.winner)} ${verb(finished.winner, "lead")} next.`)
      : h("div", { class: "trick-outcome punished" },
        h("strong", {}, `${name(finished.cutter)} punished ${name(finished.picker)}`),
        ` · ${name(finished.picker)} ${verb(finished.picker, "pick")} up ${finished.cards.length} card${finished.cards.length === 1 ? "" : "s"}`
        + ` and ${verb(finished.picker, "lead")} next.`);

  const myHand = Array.isArray(v.hands[me]) ? v.hands[me] : [];
  const hand = myHand.map((card) => cardEl(card, {
    onClick: live ? () => ctx.act(card.id) : null,
    disabled: !legal.has(card.id),
    action: live && legal.has(card.id) ? card.id : null,
  }));
  const ledSuit = v.led_suit != null ? suitSymbol(SUIT_ORDER[v.led_suit]) : null;
  const ask = !v.trick.length ? (v.tricks_played === 0 ? "Lead the A♠." : "Lead any card.")
    : legal.size && myHand.some((c) => legal.has(c.id) && c.suit === SUIT_ORDER[v.led_suit])
      ? `Follow ${ledSuit}.` : `No ${ledSuit}: cut with any card. The highest ${ledSuit} picks up the trick.`;

  const result = !v.done ? null
    : v.donkey != null ? `${ctx.seatName(v.donkey)} ${v.donkey === ctx.board.human_seat ? "are" : "is"} the Donkey!${v.stopped ? " (game stopped: repeated position, most cards loses)" : ""}`
    : v.losers.length ? `Stopped with a tie for most cards: ${v.losers.map((s) => ctx.seatName(s)).join(" & ")}`
    : "Everyone ran out together: no Donkey";

  root.append(h("div", { class: "felt table4 donkey" },
    ...seats,
    h("div", { class: "seat center" }, trick,
      outcome,
      h("div", { class: "note" }, `Discarded ${v.discarded} · trick ${v.tricks_played + (finished ? 0 : 1)}`)),
    h("div", { class: "seat south" },
      seatTag(ctx, me, status(me)),
      prompt(ctx, ask),
      Array.isArray(v.hands[me]) ? h("div", { class: "hand wrap fan" }, hand) : otherHand(v.hands[me]),
      result ? h("div", { class: `result ${v.donkey === ctx.board.human_seat ? "loss" : ""}` }, result) : null)));
}

/** Extra pause (ms) before the next computer move while a finished trick is on show. */
export function holdAfter(board) {
  const v = board.view;
  return !v.done && !v.trick.length && v.last_trick ? 1600 : 0;
}
