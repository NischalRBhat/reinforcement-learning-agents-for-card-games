// Canasta: you and your partner (north) against west and east. Each side's melds are shown
// in a strip; the pile shows whether it is frozen or blocked. To play, click a card in your
// hand: the moves that card allows (melds, adds, discard) appear underneath.
import { cardEl, h } from "../cards.js";
import { actionBar, named, otherHand, position, POSITIONS_4, prompt, seatTag, viewerSeat } from "./common.js";

const JOKER = 0;
const TWO = 1;
const selections = new Map(); // `${session}:${board}` -> selected kind

function meldStrip(ctx, v, team, title) {
  const melds = v.melds[team];
  const opening = v.team_melded[team] ? null : `opening meld needs ${v.minimums[team]}`;
  return h("div", { class: `meld-area ${team === viewerSeat(ctx.board) % 2 ? "ours" : "theirs"}` },
    h("div", { class: "meld-title" }, h("strong", {}, title),
      ` · ${v.canastas[team]} canasta${v.canastas[team] === 1 ? "" : "s"}`, opening ? ` · ${opening}` : ""),
    melds.length
      ? h("div", { class: "melds" }, melds.map((m) =>
        h("div", { class: `meld ${m.canasta ? `canasta ${m.canasta}` : ""}`, title: `${m.naturals} natural, ${m.wilds} wild` },
          h("div", { class: "hand compact", style: "--step:9px" }, m.cards.map((c) => cardEl(c, { small: true }))),
          h("span", { class: "meld-label" }, `${m.plural} ×${m.size}`, m.canasta ? h("span", { class: "canasta-badge" }, m.canasta) : null))))
      : h("div", { class: "note" }, "No melds yet"));
}

function redThrees(v, seat) {
  return v.red_threes[seat].length
    ? h("div", { class: "red-threes", title: "Red threes: +100 each" }, v.red_threes[seat].map((c) => cardEl(c, { small: true })))
    : null;
}

function pileZone(v) {
  const badges = [v.pile_frozen ? h("span", { class: "pill warn" }, "Frozen") : null,
    v.pile_blocked ? h("span", { class: "pill" }, "Blocked by a black 3") : null];
  return h("div", { class: "pile-row" },
    h("div", { class: "zone" }, h("div", { class: "pile" }, v.stock_count ? cardEl(null) : h("div", { class: "slot" }),
      h("span", { class: "pile-count" }, `Stock ${v.stock_count}`))),
    h("div", { class: "zone" }, h("div", { class: "pile" }, v.pile_top ? cardEl(v.pile_top) : h("div", { class: "slot" }),
      h("span", { class: "pile-count" }, `Pile ${v.pile_count}`)), h("div", { class: "badges" }, badges)));
}

/** Legal moves that involve cards of ``kind``. */
function movesFor(legal, kind) {
  return legal.filter((a) => (a.type === "discard" && a.kind === kind)
    || (a.type === "meld" && (a.kind === kind && a.naturals > 0
      || (kind === JOKER && a.jokers > 0) || (kind === TWO && a.twos > 0))));
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const me = viewerSeat(ctx.board);
  const myTeam = me % 2;
  const partner = (me + 2) % 4;
  const live = ctx.board.human_turn;
  const legal = v.legal_actions;
  const key = `${ctx.session.id}:${ctx.index}`;
  const myHand = Array.isArray(v.hands[me]) ? v.hands[me] : [];

  // The selected kind: keep it while it still has moves, else pick the first that does.
  let selected = selections.get(key);
  if (live && v.phase === "play" && !(selected != null && movesFor(legal, selected).length)) {
    selected = myHand.map((c) => c.kind).find((k) => movesFor(legal, k).length) ?? null;
  }
  selections.set(key, selected);

  const seats = [0, 1, 2, 3].filter((s) => s !== me).map((seat) =>
    h("div", { class: `seat ${POSITIONS_4[position(ctx.board, seat)]}` },
      seatTag(ctx, seat, seat === partner ? "partner" : "opponent"),
      otherHand(v.hands[seat]), redThrees(v, seat)));

  const hand = myHand.map((card) => cardEl(card, {
    onClick: live && v.phase === "play" ? () => { selections.set(key, card.kind); ctx.rerender(); } : null,
    selected: card.kind === selected,
    title: `${card.label}${card.kind === JOKER || card.kind === TWO ? " (wild)" : ""}`,
  }));

  let controls = null;
  if (live) {
    const opening = !v.team_melded[myTeam] && (v.turn_value > 0 || v.pile_pending)
      ? h("div", { class: "note" }, `Opening meld: ${v.turn_value} of ${v.minimums[myTeam]} points so far`
        + (v.pile_pending ? ` · the other ${v.pile_pending} pile cards come once you reach it` : "")) : null;
    if (v.phase === "draw") {
      controls = [prompt(ctx, "Draw from the stock, or take the pile by melding its top card."), actionBar(ctx)];
    } else {
      const options = selected != null ? movesFor(legal, selected) : [];
      controls = [
        prompt(ctx, "Click a card to see what you can meld or discard with it. A discard ends your turn."),
        opening,
        options.length ? actionBar(ctx, options, "kind-moves") : null,
        h("details", { class: "legal-moves" }, h("summary", {}, `All legal moves (${legal.length})`), actionBar(ctx)),
      ];
    }
  }

  const result = v.done && v.hand_scores
    ? h("div", { class: "result" }, `${v.team_names[myTeam]} ${v.hand_scores[myTeam]} · ${v.team_names[1 - myTeam]} ${v.hand_scores[1 - myTeam]}`
      + (v.went_out != null ? ` · ${ctx.seatName(v.went_out)} went out${v.concealed ? " concealed" : ""}` : " · the stock ran out"))
    : null;

  root.append(h("div", { class: "felt table4 canasta" },
    ...seats,
    h("div", { class: "meld-row theirs-row" }, meldStrip(ctx, v, 1 - myTeam, `Opponents (${v.team_names[1 - myTeam]})`)),
    h("div", { class: "seat center" }, pileZone(v), v.last_event ? h("div", { class: "note" }, named(ctx, v.last_event)) : null),
    h("div", { class: "meld-row ours-row" }, meldStrip(ctx, v, myTeam, `Your side (${v.team_names[myTeam]})`)),
    h("div", { class: "seat south" },
      seatTag(ctx, me, `${v.hand_counts[me]} cards`),
      redThrees(v, me),
      controls,
      Array.isArray(v.hands[me]) ? h("div", { class: "hand wrap fan" }, hand) : otherHand(v.hands[me]),
      result)));
}

/** A short pause after a pile pickup or a meld so the table can be read. */
export function holdAfter(board) {
  const event = board.view.last_event ?? "";
  return !board.view.done && /took the pile|went out/.test(event) ? 1200 : 0;
}
