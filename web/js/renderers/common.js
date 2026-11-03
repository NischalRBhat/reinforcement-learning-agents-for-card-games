// Pieces shared by the game renderers.
import { cardEl, h } from "../cards.js";

export const AVATARS = { human: "You", agent: "AI", system: "S", random: "R" };

/** Buttons for legal actions (only on the human's turn). */
export function actionBar(ctx, legal = ctx.board.view.legal_actions, className = "") {
  if (!ctx.board.human_turn || !legal?.length) return null;
  return h("div", { class: `actions ${className}` }, legal.map((a) =>
    h("button", { class: "act", type: "button", dataset: { action: a.action }, onClick: () => ctx.act(a.action) }, a.label)));
}

/** The seat drawn at the bottom: the human's, else seat 0. */
export function viewerSeat(board) {
  return board.human_seat ?? 0;
}

/** Table position (0 bottom, then clockwise) of a seat relative to the viewer. */
export function position(board, seat) {
  const n = board.seats.length;
  return (seat - viewerSeat(board) + n) % n;
}

export const POSITIONS_4 = ["south", "west", "north", "east"];

/** A hand that may be hidden (a count) or revealed (a card list). Long hands overlap more to fit. */
export function otherHand(hand, max = 13) {
  if (Array.isArray(hand)) {
    const step = Math.max(7, Math.min(14, Math.floor(200 / Math.max(1, hand.length))));
    return h("div", { class: "hand compact", style: `--step:${step}px` }, hand.map((c) => cardEl(c, { small: true })));
  }
  const count = hand ?? 0;
  return h("div", { class: "hand compact backs" },
    Array.from({ length: Math.min(count, max) }, () => cardEl(null, { small: true })),
    h("span", { class: "count" }, `${count} card${count === 1 ? "" : "s"}`));
}

/** Name plate for a seat: avatar coloured by controller, turn and winner states, extra info. */
export function seatTag(ctx, seat, ...extra) {
  const board = ctx.board;
  const controller = board.seats[seat];
  const turn = !board.done && board.current_player === seat;
  const winner = board.done && board.winners.includes(seat) && board.seats.length > 1;
  const thinking = turn && controller !== "human";
  const info = extra.flat().filter((x) => x != null && x !== "").map((x) => (typeof x === "string" ? x.replace(/^ · /, "") : x));
  return h("div", { class: `seat-tag ctl-${controller} ${turn ? "turn" : ""} ${winner ? "winner" : ""}` },
    h("span", { class: "avatar", "aria-hidden": "true" }, AVATARS[controller] ?? "?"),
    h("span", { class: "seat-name" }, ctx.seatName(seat)),
    info.map((x) => h("span", { class: "seat-info" }, x)),
    thinking ? h("span", { class: "thinking", "aria-label": "thinking" }, h("i"), h("i"), h("i")) : null,
    winner ? h("span", { class: "crown", "aria-label": "winner" }, "★") : null);
}

export function prompt(ctx, text) {
  return ctx.board.human_turn ? h("div", { class: "prompt" }, h("span", { class: "prompt-dot", "aria-hidden": "true" }), text) : null;
}

/** Replace the engines' "P1".."P4" in log text with the seat names shown on screen. */
export function named(ctx, text) {
  return text ? text.replace(/\bP([1-4])\b/g, (_, n) => ctx.seatName(Number(n) - 1)) : text;
}

/** A result banner on the felt: tone is "win", "loss" or "". */
export function resultPill(text, tone = "") {
  return h("div", { class: `result ${tone}` }, text);
}
