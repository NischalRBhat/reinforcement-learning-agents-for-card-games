// Two-player rummy: draw from the stock or take the discard, then click a card to discard.
import { cardEl, h } from "../cards.js";
import { named, otherHand, prompt, seatTag, viewerSeat } from "./common.js";

const DRAW_STOCK = 0;
const DRAW_DISCARD = 1;
const DISCARD = 2;

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const live = ctx.board.human_turn;
  const legal = new Set(v.legal_actions.map((a) => a.action));
  const myHand = Array.isArray(v.hands[me]) ? v.hands[me] : [];
  const deadwood = (seat) => (v.final_deadwood ? ` · deadwood ${v.final_deadwood[seat]}` : "");

  const stock = h("div", { class: "pile" },
    cardEl(null, {
      onClick: live && legal.has(DRAW_STOCK) ? () => ctx.act(DRAW_STOCK) : null,
      action: legal.has(DRAW_STOCK) ? DRAW_STOCK : null,
      title: "Draw from the stock",
    }),
    h("span", { class: "pile-count" }, `Stock ${v.stock_count}`));
  const discard = h("div", { class: "pile" },
    v.discard_top
      ? cardEl(v.discard_top, {
        onClick: live && legal.has(DRAW_DISCARD) ? () => ctx.act(DRAW_DISCARD) : null,
        action: legal.has(DRAW_DISCARD) ? DRAW_DISCARD : null,
        highlight: live && legal.has(DRAW_DISCARD),
        title: `Take ${v.discard_top.label}`,
      })
      : h("div", { class: "slot" }),
    h("span", { class: "pile-count" }, `Discard ${v.discard_count}`));

  const hand = myHand.map((card) => {
    const action = DISCARD + card.id;
    const canDiscard = live && legal.has(action);
    return cardEl(card, {
      onClick: live && v.phase === "discard" ? () => ctx.act(action) : null,
      disabled: !canDiscard,
      action: canDiscard ? action : null,
      highlight: v.drawn_from_discard?.id === card.id,
      title: canDiscard ? `Discard ${card.label}` : card.label,
    });
  });

  const outcome = v.done
    ? h("div", { class: "result" }, v.winner != null ? `${ctx.seatName(v.winner)} went out` : "Stock ran out: lower deadwood wins")
    : null;

  root.append(h("div", { class: "felt rummy" },
    h("div", { class: "opponent" }, seatTag(ctx, 1 - me, deadwood(1 - me)), otherHand(v.hands[1 - me])),
    h("div", { class: "center-row" }, stock, discard),
    v.last_event ? h("div", { class: "note" }, named(ctx, v.last_event)) : null,
    prompt(ctx, v.phase === "draw" ? "Draw from the stock or take the discard." : "Click a card to discard. You go out when the rest all meld."),
    h("div", { class: "me" },
      seatTag(ctx, me, v.deadwood != null && !v.done ? ` · deadwood ${v.deadwood}` : deadwood(me)),
      Array.isArray(v.hands[me]) ? h("div", { class: "hand wrap fan" }, hand) : otherHand(v.hands[me])),
    outcome));
}
