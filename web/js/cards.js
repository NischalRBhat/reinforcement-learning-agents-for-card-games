// DOM helpers and the card renderer shared by every game.

const SUIT_SYMBOLS = { C: "♣", D: "♦", H: "♥", S: "♠" };
const SUIT_NAMES = { C: "clubs", D: "diamonds", H: "hearts", S: "spades" };
const RANK_NAMES = { A: "Ace", J: "Jack", Q: "Queen", K: "King" };
export const SUIT_ORDER = ["C", "D", "H", "S"];

export function suitSymbol(suit) {
  return SUIT_SYMBOLS[suit] ?? suit;
}

/** h("div", {class: "x", onClick: fn, dataset: {a: 1}}, child, "text", [more]) */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(props ?? {})) {
    if (value == null || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key === "dataset") {
      for (const [k, v] of Object.entries(value)) if (v != null) el.dataset[k] = v;
    } else if (key.startsWith("on") && typeof value === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

function spokenName(card) {
  return `${RANK_NAMES[card.rank] ?? card.rank} of ${SUIT_NAMES[card.suit] ?? card.suit}`;
}

/**
 * A card face (or back when ``card`` is null).
 * opts: small, onClick, disabled, selected, highlight, className, action (data-action), title
 */
export function cardEl(card, opts = {}) {
  const classes = ["card"];
  if (opts.small) classes.push("small");
  if (opts.selected) classes.push("selected");
  if (opts.highlight) classes.push("highlight");
  if (opts.className) classes.push(opts.className);
  if (!card) classes.push("back");
  else {
    classes.push(card.suit === "D" || card.suit === "H" ? "red" : "black");
    if ("JQK".includes(card.rank)) classes.push("face");
  }
  const clickable = typeof opts.onClick === "function";
  if (clickable) classes.push("clickable");
  const el = h(clickable ? "button" : "div", {
    class: classes.join(" "),
    type: clickable ? "button" : null,
    title: opts.title ?? card?.label,
    "aria-label": card ? spokenName(card) : "face-down card",
    "aria-pressed": clickable && opts.selected ? "true" : null,
    disabled: clickable && opts.disabled,
    onClick: clickable ? opts.onClick : null,
    dataset: { card: card?.id, action: opts.action },
  });
  if (card) {
    const symbol = suitSymbol(card.suit);
    const index = (pos) => h("span", { class: `idx ${pos}`, "aria-hidden": "true" },
      h("span", { class: "r" }, card.rank), h("span", { class: "s" }, symbol));
    el.append(
      index("tl"),
      h("span", { class: "center", "aria-hidden": "true" },
        "JQK".includes(card.rank) ? [h("span", { class: "face-letter" }, card.rank), h("span", { class: "face-suit" }, symbol)] : symbol),
      index("br"),
    );
  }
  return el;
}

export function fmtSigned(value, digits = 2) {
  if (value == null) return "–";
  let rounded = Number(value).toFixed(digits);
  if (digits > 0) rounded = rounded.replace(/\.?0+$/, "");
  return value > 0 ? `+${rounded}` : rounded;
}
