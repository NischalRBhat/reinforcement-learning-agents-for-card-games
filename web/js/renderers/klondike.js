// Klondike: click a card (or run) to pick it up, then click where it should go.
// Clicking a picked-up card again sends it to the foundation when that is legal.
import { cardEl, h, SUIT_ORDER, suitSymbol } from "../cards.js";
import { actionBar } from "./common.js";

const selections = new Map(); // `${session}:${board}` -> source

function matchesSource(action, source) {
  if (!source || action.from !== source.from) return false;
  if (source.from === "tableau") return action.src === source.src && action.n === source.n;
  if (source.from === "foundation") return action.suit === source.suit;
  return source.from === "waste";
}

function matchesTarget(action, target) {
  return action.to === target.to && (target.to !== "tableau" || action.dst === target.dst);
}

function sameSource(a, b) {
  return a && b && a.from === b.from && a.src === b.src && a.n === b.n && a.suit === b.suit;
}

export function render(root, ctx) {
  const v = ctx.board.view;
  const key = `${ctx.session.id}:${ctx.index}`;
  const legal = v.legal_actions;
  const live = ctx.board.human_turn;
  let selected = live ? selections.get(key) : null;
  if (selected && !legal.some((a) => matchesSource(a, selected))) selected = null;
  selections.set(key, selected);

  const moves = selected ? legal.filter((a) => matchesSource(a, selected)) : [];
  const play = (action) => { selections.set(key, null); ctx.act(action.action); };
  const choose = (source) => { selections.set(key, source); ctx.rerender(); };
  const tryTarget = (target) => {
    const move = moves.find((a) => matchesTarget(a, target));
    if (move) play(move);
    return Boolean(move);
  };
  const pick = (source) => {
    if (!live) return;
    if (sameSource(selected, source)) {
      const home = legal.find((a) => matchesSource(a, source) && a.to === "foundation");
      return home ? play(home) : choose(null);
    }
    if (legal.some((a) => matchesSource(a, source))) choose(source);
    else if (selected) choose(null);
  };
  const isTarget = (target) => moves.some((a) => matchesTarget(a, target));

  // Stock and waste
  const stockAction = legal.find((a) => a.from === "stock");
  const stock = h("div", { class: "pile stock" },
    v.stock_count
      ? cardEl(null, { onClick: live && stockAction ? () => ctx.act(stockAction.action) : null, action: stockAction?.action, title: "Draw" })
      : h("button", { class: "slot recycle", disabled: !(live && stockAction), dataset: { action: stockAction?.action },
          onClick: () => stockAction && ctx.act(stockAction.action) }, "↺"),
    h("span", { class: "pile-count" }, `${v.stock_count}`));
  const wasteSource = { from: "waste" };
  const waste = h("div", { class: "pile waste" },
    v.waste.map((card, i) => {
      const top = i === v.waste.length - 1;
      return cardEl(card, {
        className: "fan",
        selected: top && sameSource(selected, wasteSource),
        onClick: top && live ? () => pick(wasteSource) : null,
      });
    }),
    v.waste.length ? null : h("div", { class: "slot" }));

  // Foundations, one per suit
  const foundations = h("div", { class: "foundations" }, SUIT_ORDER.map((suit, s) => {
    const card = v.foundation[s];
    const source = { from: "foundation", suit: s };
    const onClick = live ? () => (selected ? tryTarget({ to: "foundation" }) || pick(source) : pick(source)) : null;
    const target = isTarget({ to: "foundation" }) && (!selected || moves.some((a) => a.to === "foundation"));
    return card
      ? cardEl(card, { onClick, selected: sameSource(selected, source), highlight: target })
      : h("button", { class: `slot ${target ? "highlight" : ""}`, disabled: !live, onClick }, suitSymbol(suit));
  }));

  // Tableau
  const tableau = h("div", { class: "tableau" }, v.tableau.map((pile, col) => {
    const target = { to: "tableau", dst: col };
    const column = h("div", {
      class: `column ${isTarget(target) ? "highlight" : ""}`,
      onClick: live ? () => selected && tryTarget(target) : null,
    });
    if (!pile.length) column.append(h("div", { class: "slot" }));
    pile.forEach((card, i) => {
      if (!card) return column.append(cardEl(null, { className: "stacked" }));
      const source = { from: "tableau", src: col, n: pile.length - i };
      const inSelection = selected?.from === "tableau" && selected.src === col && i >= pile.length - selected.n;
      column.append(cardEl(card, {
        className: "stacked",
        selected: inSelection,
        onClick: live ? (event) => {
          event.stopPropagation();
          if (selected && selected.src !== col && tryTarget(target)) return;
          pick(source);
        } : null,
      }));
    });
    return column;
  }));

  const ending = v.won ? "Solved!" : v.deadlocked
    ? `Deadlocked: no sequence of moves can make progress (${v.foundation_count}/52 home)`
    : `Stopped at the move limit: ${v.foundation_count}/52 cards home`;
  const status = v.done
    ? h("div", { class: `result ${v.won ? "win" : "loss"}` }, ending)
    : h("div", { class: "klondike-status" },
      [["Home", `${v.foundation_count}/52`], ["Moves", v.steps], ["Stock", v.stock_count], ["Waste", v.waste_count]]
        .map(([k, value]) => h("span", { class: "stat-chip" }, `${k} `, h("strong", {}, value))));
  const solved = ctx.board.solve_result;
  const verdicts = {
    winnable: "This deal was winnable (the solver found a win with every card visible).",
    "not winnable": "This deal could not be won, even with every card visible.",
    unknown: "The solver couldn't decide within its search budget (usually a deal that can't be won).",
  };
  const tools = h("div", { class: "actions" },
    ctx.board.undo_available ? h("button", { class: "act", type: "button", id: `undo-${ctx.index}`, onClick: ctx.undo, title: "Take back your last move" }, "↶ Undo") : null,
    v.done && !v.won && !solved
      ? h("button", { class: "act", id: `solve-${ctx.index}`, onClick: ctx.solve }, "Was this deal winnable?") : null);

  root.append(h("div", { class: "felt klondike" },
    h("div", { class: "top-row" }, h("div", { class: "stock-waste" }, stock, waste), foundations),
    tableau,
    status,
    solved ? h("div", { class: "note solver" }, verdicts[solved.verdict]) : null,
    tools.childElementCount ? tools : null,
    live && legal.length
      ? h("details", { class: "legal-moves" }, h("summary", {}, `All legal moves (${legal.length})`), actionBar(ctx))
      : null));
}
