// Heads-up hold'em: opponent on top, you at the bottom, each player's latest action in a
// bubble, and a headline that says how the hand ended.
import { cardEl, fmtSigned, h } from "../cards.js";
import { actionBar, actionBubble, named, prompt, seatTag, viewerSeat } from "./common.js";

const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);

function seatBox(ctx, seat) {
  const v = ctx.board.view;
  const hole = v.hole[seat];
  const folded = v.folded === seat;
  const cards = hole ? hole.map((c) => cardEl(c)) : [cardEl(null), cardEl(null)];
  const last = v.last_actions?.[seat];
  return h("div", { class: `poker-seat ${folded ? "folded" : ""}` },
    seatTag(ctx, seat, v.button === seat ? h("span", { class: "dealer-button", title: "Button (small blind)" }, "D") : null),
    actionBubble(last ? capitalize(last.text) : null, last?.current),
    h("div", { class: "hand" }, cards),
    h("div", { class: "chips" }, "Stack ", h("strong", {}, v.stacks[seat]),
      v.street_bets[seat] ? h("span", { class: "bet-chip" }, ` · bet ${v.street_bets[seat]}`) : null),
    v.hand_names?.[seat] ? h("div", { class: "note" }, v.hand_names[seat]) : null,
    v.result ? h("div", { class: `result ${v.result[seat] > 0 ? "win" : v.result[seat] < 0 ? "loss" : ""}` },
      `${fmtSigned(v.result[seat], 0)} chips`) : null);
}

function ending(ctx) {
  const v = ctx.board.view;
  if (!v.done) return null;
  const name = (seat) => ctx.seatName(seat);
  const verb = (seat) => (seat === ctx.board.human_seat ? "win" : "wins");
  if (v.folded != null) {
    const winner = 1 - v.folded;
    const when = v.street === "preflop" ? "before the flop" : `on the ${v.street}`;
    return h("div", { class: "hand-ending" },
      h("strong", {}, `${name(v.folded)} folded ${when}`),
      ` · ${name(winner)} ${verb(winner)} the pot without a showdown (+${v.result[winner]} chips)`);
  }
  if (!v.result[0]) return h("div", { class: "hand-ending" }, h("strong", {}, "Showdown: split pot"), ` · both hold ${v.hand_names[0]}`);
  const winner = v.result[0] > 0 ? 0 : 1;
  return h("div", { class: "hand-ending" },
    h("strong", {}, `Showdown: ${name(winner)} ${verb(winner)} with ${v.hand_names[winner]}`),
    ` · against ${v.hand_names[1 - winner]} (+${v.result[winner]} chips)`);
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const board = Array.from({ length: 5 }, (_, i) => (v.board[i] ? cardEl(v.board[i]) : h("div", { class: "slot" })));
  root.append(h("div", { class: "felt poker" },
    seatBox(ctx, 1 - me),
    h("div", { class: "poker-center" },
      h("div", { class: "hand board-cards" }, board),
      h("div", { class: "pot" }, h("span", { class: "chip-ico", "aria-hidden": "true" }), `Pot ${v.pot}`,
        h("span", { class: "street" }, v.street)),
      ending(ctx),
      v.to_call && ctx.board.human_turn ? prompt(ctx, `${v.to_call} to call`) : null),
    seatBox(ctx, me),
    actionBar(ctx),
    v.log?.length ? h("div", { class: "table-log" }, v.log.slice(-4).map((line) => named(ctx, line)).join(" · ")) : null));
}

/** Pause when new board cards have just been dealt, so they are seen before the next bet. */
export function holdAfter(board) {
  const v = board.view;
  return !v.done && v.board.length && !(v.last_actions ?? []).some((a) => a?.current) ? 900 : 0;
}
