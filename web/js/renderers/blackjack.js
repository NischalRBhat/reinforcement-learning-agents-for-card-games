// Blackjack: dealer on top, the player's hand(s) below, every decision shown as it is made.
import { cardEl, fmtSigned, h } from "../cards.js";
import { actionBar, moveTrail, prompt } from "./common.js";

function dealerSummary(v) {
  if (!v.done) return null;
  const drew = v.dealer.cards.length - 2;
  const busts = v.dealer.total > 21;
  const text = busts ? `Dealer drew ${drew} card${drew === 1 ? "" : "s"} and busts with ${v.dealer.total}`
    : drew > 0 ? `Dealer drew ${drew} card${drew === 1 ? "" : "s"} to ${v.dealer.total}`
    : `Dealer stands on ${v.dealer.total}`;
  return h("div", { class: `note dealer-note ${busts ? "bust" : ""}` }, text);
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const result = v.done
    ? h("div", { class: `result ${v.net > 0 ? "win" : v.net < 0 ? "loss" : ""}` },
      v.net > 0 ? `Won ${fmtSigned(v.net)} units` : v.net < 0 ? `Lost ${fmtSigned(v.net)} units` : "Push: no units won or lost")
    : null;
  const moves = ctx.board.moves;
  root.append(h("div", { class: "felt blackjack" },
    h("div", { class: "zone dealer" },
      h("div", { class: "zone-label" }, `Dealer · ${v.dealer.total}${v.done ? "" : " showing"}`),
      h("div", { class: "hand" }, v.dealer.cards.map((c) => cardEl(c))),
      dealerSummary(v)),
    h("div", { class: "bj-hands" }, v.hands.map((hand, i) =>
      h("div", { class: `zone bj-hand ${hand.active ? "active" : ""}` },
        h("div", { class: "zone-label" },
          `${v.hands.length > 1 ? `Hand ${i + 1} · ` : ""}${hand.soft ? "soft " : ""}${hand.total} · bet ${hand.wager}`),
        h("div", { class: "hand" }, hand.cards.map((c) => cardEl(c))),
        hand.outcome ? h("div", { class: "outcome" }, hand.outcome) : null))),
    moves.length ? moveTrail(ctx, moves)
      : v.done ? h("div", { class: "note" }, "Settled on the deal: no decisions this hand") : null,
    v.phase === "insurance" ? prompt(ctx, "Dealer shows an ace: take insurance?") : null,
    v.insurance ? h("div", { class: "note" }, `Insurance bet: ${v.insurance}`) : null,
    result,
    actionBar(ctx)));
}

/** A short pause after each computer decision so it can be seen before the next one. */
export function holdAfter(board) {
  return board.moves.length && !board.view.done ? 500 : 0;
}
