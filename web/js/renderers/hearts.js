// Hearts: pass three cards, then follow suit; legal cards are clickable, others dimmed.
// After the exchange the table shows who passed to whom, and a finished trick stays on
// the table (with who took it and for how many points) until the next lead.
import { cardEl, h } from "../cards.js";
import { otherHand, position, POSITIONS_4, prompt, seatTag, viewerSeat } from "./common.js";

const trickPoints = (plays) =>
  plays.reduce((sum, { card }) => sum + (card.suit === "H" ? 1 : card.suit === "S" && card.rank === "Q" ? 13 : 0), 0);

/** One row per pass: "Agent 2 → You", with the cards when this viewer may know them. */
function passSummary(ctx, v) {
  const name = (seat) => ctx.seatName(seat);
  return h("div", { class: "pass-summary", "aria-label": `Cards passed ${v.pass_direction}` },
    h("div", { class: "pass-title" }, `Cards passed ${v.pass_direction}`),
    v.passes.map((p) => h("div", { class: "pass-row" },
      h("span", { class: "pass-who" }, `${name(p.from)} → ${name(p.to)}`),
      h("span", { class: "hand compact" }, p.cards
        ? p.cards.map((c) => cardEl(c, { small: true }))
        : Array.from({ length: p.count }, () => cardEl(null, { small: true }))))));
}

/** One line for the player's own exchange, kept for the whole first trick. */
function myExchange(ctx, v, me) {
  const gave = v.passes?.find((p) => p.from === me && p.cards);
  const got = v.passes?.find((p) => p.to === me && p.cards);
  if (!gave || !got) return null;
  const cards = (list) => list.map((c) => c.label).join(" ");
  return h("div", { class: "note" },
    `You passed ${cards(gave.cards)} to ${ctx.seatName(gave.to)} · received ${cards(got.cards)} from ${ctx.seatName(got.from)} (outlined)`);
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const live = ctx.board.human_turn;
  const legal = new Set(v.legal_actions.map((a) => a.action));
  const match = ctx.board.stats.points;
  const points = (seat) => ` · ${v.done ? v.hand_points[seat] : v.points[seat]} pts · match ${match[seat]}`;

  const seats = [0, 1, 2, 3].filter((s) => s !== me).map((seat) =>
    h("div", { class: `seat ${POSITIONS_4[position(ctx.board, seat)]}` }, seatTag(ctx, seat, points(seat)), otherHand(v.hands[seat])));

  // Until the next card is led, the finished trick stays on the table.
  const finished = !v.trick.length && v.last_trick.length ? v.last_trick : null;
  const plays = finished ?? v.trick;
  const trick = h("div", { class: `trick ${finished ? "resolved" : ""}` }, plays.map(({ player, card }) =>
    h("div", { class: `trick-card ${POSITIONS_4[position(ctx.board, player)]}` },
      cardEl(card, { className: finished && player === v.last_winner ? "picker" : "" }))));
  const taken = finished ? trickPoints(finished) : 0;
  const outcome = finished
    ? h("div", { class: `trick-outcome ${taken ? "punished" : "cleared"}` },
      h("strong", {}, `${ctx.seatName(v.last_winner)} ${v.last_winner === ctx.board.human_seat ? "take" : "takes"} the trick`),
      taken ? ` · +${taken} point${taken === 1 ? "" : "s"}` : " · no points")
    : null;

  const firstTrick = v.phase === "play" && v.tricks_played === 0;
  const myHand = Array.isArray(v.hands[me]) ? v.hands[me] : [];
  const picks = new Set((v.pass_picks ?? []).map((c) => c.id));
  const received = new Set(firstTrick ? (v.received ?? []).map((c) => c.id) : []);
  const hand = myHand.map((card) => cardEl(card, {
    onClick: live ? () => ctx.act(card.id) : null,
    disabled: !legal.has(card.id),
    action: live && legal.has(card.id) ? card.id : null,
    selected: picks.has(card.id),
    className: received.has(card.id) ? "received" : "",
    title: received.has(card.id) ? `${card.label} (passed to you)` : null,
  }));

  const info = [];
  if (v.phase === "passing") info.push(`Passing ${v.pass_direction}`);
  if (v.hearts_broken) info.push("hearts broken");
  if (v.done && v.moon_shooter != null) info.push(`${ctx.seatName(v.moon_shooter)} shot the moon!`);

  root.append(h("div", { class: "felt table4 hearts" },
    ...seats,
    h("div", { class: "seat center" },
      // Before the first lead the exchange fills the empty trick spot; then the trick takes over.
      firstTrick && v.passes && !v.trick.length ? passSummary(ctx, v) : [trick, outcome],
      info.length ? h("div", { class: "note" }, info.join(" · ")) : null),
    h("div", { class: "seat south" },
      seatTag(ctx, me, points(me)),
      firstTrick ? myExchange(ctx, v, me) : null,
      prompt(ctx, v.phase === "passing" ? `Choose 3 cards to pass ${v.pass_direction} (${picks.size}/3)` : "Play a card"),
      Array.isArray(v.hands[me]) ? h("div", { class: "hand wrap fan" }, hand) : otherHand(v.hands[me]),
      v.done ? h("div", { class: "result" }, `Hand points: ${v.hand_points.map((p, s) => `${ctx.seatName(s)} ${p}`).join(", ")}`) : null)));
}

/** Pauses so the exchange and each finished trick can be read before play continues. */
export function holdAfter(board) {
  const v = board.view;
  if (v.done || v.trick.length) return 0;
  if (v.phase === "play" && v.tricks_played === 0 && v.passes) return 2500;  // the pass just happened
  return v.last_trick.length ? 1400 : 0;
}
