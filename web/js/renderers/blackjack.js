import { cardEl, fmtSigned, h } from "../cards.js";
import { actionBar, prompt } from "./common.js";

export function render(root, ctx) {
  const v = ctx.board.view;
  const result = v.done
    ? h("div", { class: `result ${v.net > 0 ? "win" : v.net < 0 ? "loss" : ""}` },
      v.net > 0 ? `Won ${fmtSigned(v.net)} units` : v.net < 0 ? `Lost ${fmtSigned(v.net)} units` : "Push: no units won or lost")
    : null;
  root.append(h("div", { class: "felt blackjack" },
    h("div", { class: "zone dealer" },
      h("div", { class: "zone-label" }, `Dealer · ${v.dealer.total}${v.done ? "" : " showing"}`),
      h("div", { class: "hand" }, v.dealer.cards.map((c) => cardEl(c)))),
    h("div", { class: "bj-hands" }, v.hands.map((hand, i) =>
      h("div", { class: `zone bj-hand ${hand.active ? "active" : ""}` },
        h("div", { class: "zone-label" },
          `${v.hands.length > 1 ? `Hand ${i + 1} · ` : ""}${hand.soft ? "soft " : ""}${hand.total} · bet ${hand.wager}`),
        h("div", { class: "hand" }, hand.cards.map((c) => cardEl(c))),
        hand.outcome ? h("div", { class: "outcome" }, hand.outcome) : null))),
    v.phase === "insurance" ? prompt(ctx, "Dealer shows an ace: take insurance?") : null,
    v.insurance ? h("div", { class: "note" }, `Insurance bet: ${v.insurance}`) : null,
    result,
    actionBar(ctx)));
}
