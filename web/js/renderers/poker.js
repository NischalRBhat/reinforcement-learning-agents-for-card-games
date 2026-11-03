// Heads-up hold'em: opponent on top, you at the bottom, bet buttons with chip amounts.
import { cardEl, fmtSigned, h } from "../cards.js";
import { actionBar, named, prompt, seatTag, viewerSeat } from "./common.js";

function seatBox(ctx, seat) {
  const v = ctx.board.view;
  const hole = v.hole[seat];
  const cards = hole ? hole.map((c) => cardEl(c)) : v.folded === seat ? [] : [cardEl(null), cardEl(null)];
  return h("div", { class: `poker-seat ${v.folded === seat ? "folded" : ""}` },
    seatTag(ctx, seat, v.button === seat ? h("span", { class: "dealer-button", title: "Button (small blind)" }, "D") : null),
    h("div", { class: "hand" }, cards),
    h("div", { class: "chips" }, "Stack ", h("strong", {}, v.stacks[seat]),
      v.street_bets[seat] ? h("span", { class: "bet-chip" }, ` · bet ${v.street_bets[seat]}`) : null),
    v.hand_names?.[seat] ? h("div", { class: "note" }, v.hand_names[seat]) : null,
    v.result ? h("div", { class: `result ${v.result[seat] > 0 ? "win" : v.result[seat] < 0 ? "loss" : ""}` },
      `${fmtSigned(v.result[seat], 0)} chips`) : null);
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
      v.to_call && ctx.board.human_turn ? prompt(ctx, `${v.to_call} to call`) : null),
    seatBox(ctx, me),
    actionBar(ctx),
    v.log?.length ? h("div", { class: "table-log" }, v.log.slice(-4).map((line) => named(ctx, line)).join(" · ")) : null));
}
