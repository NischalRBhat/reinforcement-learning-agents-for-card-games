// Hearts: pass three cards, then follow suit; legal cards are clickable, others dimmed.
import { cardEl, h } from "../cards.js";
import { otherHand, position, POSITIONS_4, prompt, seatTag, viewerSeat } from "./common.js";

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const live = ctx.board.human_turn;
  const legal = new Set(v.legal_actions.map((a) => a.action));
  const match = ctx.board.stats.points;
  const points = (seat) => ` · ${v.done ? v.hand_points[seat] : v.points[seat]} pts · match ${match[seat]}`;

  const seats = [0, 1, 2, 3].filter((s) => s !== me).map((seat) =>
    h("div", { class: `seat ${POSITIONS_4[position(ctx.board, seat)]}` }, seatTag(ctx, seat, points(seat)), otherHand(v.hands[seat])));

  const trick = h("div", { class: "trick" }, v.trick.map(({ player, card }) =>
    h("div", { class: `trick-card ${POSITIONS_4[position(ctx.board, player)]}` }, cardEl(card))));

  const myHand = Array.isArray(v.hands[me]) ? v.hands[me] : [];
  const picks = new Set((v.pass_picks ?? []).map((c) => c.id));
  const hand = myHand.map((card) => cardEl(card, {
    onClick: live ? () => ctx.act(card.id) : null,
    disabled: !legal.has(card.id),
    action: live && legal.has(card.id) ? card.id : null,
    selected: picks.has(card.id),
  }));

  const info = [];
  if (v.phase === "passing") info.push(`Passing ${v.pass_direction}`);
  if (v.hearts_broken) info.push("hearts broken");
  if (v.last_winner != null && !v.trick.length && v.tricks_played) info.push(`${ctx.seatName(v.last_winner)} took the last trick`);
  if (v.done && v.moon_shooter != null) info.push(`${ctx.seatName(v.moon_shooter)} shot the moon!`);

  root.append(h("div", { class: "felt table4 hearts" },
    ...seats,
    h("div", { class: "seat center" }, trick, h("div", { class: "note" }, info.join(" · "))),
    h("div", { class: "seat south" },
      seatTag(ctx, me, points(me)),
      v.received?.length && v.phase !== "passing" && v.tricks_played === 0
        ? h("div", { class: "note" }, `Received: ${v.received.map((c) => c.label).join(" ")}`) : null,
      prompt(ctx, v.phase === "passing" ? `Choose 3 cards to pass ${v.pass_direction} (${picks.size}/3)` : "Play a card"),
      Array.isArray(v.hands[me]) ? h("div", { class: "hand wrap fan" }, hand) : otherHand(v.hands[me]),
      v.done ? h("div", { class: "result" }, `Hand points: ${v.hand_points.map((p, s) => `${ctx.seatName(s)} ${p}`).join(", ")}`) : null)));
}
