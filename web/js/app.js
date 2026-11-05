// App shell: game rail, setup bar, the move loop, result banner, insight, scoreboard and history.
// Game rules live in the backend; renderers only draw views and send actions.
import { api } from "./api.js";
import { fmtSigned, h } from "./cards.js";
import { GAME_META, GUIDES, RULES, SHORTCUTS } from "./guide.js";
import { AVATARS } from "./renderers/common.js";
import { RENDERERS } from "./renderers/index.js";

const SPEEDS = [1200, 700, 350, 120, 0]; // ms between computer moves
const SPEED_NAMES = ["Slow", "Relaxed", "Normal", "Fast", "Instant"];
const AUTO_NEXT_MS = 4000;
const ICONS = {
  pause: `<svg viewBox="0 0 16 16" width="14" height="14"><rect x="3" y="2" width="3.5" height="12" rx="1" fill="currentColor"/><rect x="9.5" y="2" width="3.5" height="12" rx="1" fill="currentColor"/></svg>`,
  play: `<svg viewBox="0 0 16 16" width="14" height="14"><path d="M4 2.5v11a.8.8 0 0 0 1.2.7l9-5.5a.8.8 0 0 0 0-1.4l-9-5.5A.8.8 0 0 0 4 2.5z" fill="currentColor"/></svg>`,
};
const CONTROLLER_NAMES = { human: "You", system: "System", agent: "Agent", counting: "Counting agent", random: "Random" };
const CONTROLLER_OPTIONS = { human: "You (human)", system: "System bot", agent: "Trained agent", counting: "Card-counting agent", random: "Random" };
const VARIANT_OF = { agent: "base", counting: "counting" };   // seat controller -> agent kind
const AGENT_TITLES = { base: "Trained agent", counting: "Card-counting agent" };
const METRICS = {
  ev: { label: "EV per hand", short: "EV/hand", fmt: (v) => fmtSigned(v, 3) },
  win_rate: { label: "win rate", short: "win rate", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  bb_per_100: { label: "big blinds per 100 hands", short: "bb/100", fmt: (v) => fmtSigned(v, 1) },
  mean_points: { label: "points per hand", short: "pts/hand", fmt: (v) => v.toFixed(2) },
  donkey_rate: { label: "of games as the Donkey", short: "Donkey rate", fmt: (v) => `${(v * 100).toFixed(1)}%` },
  point_diff: { label: "points per hand above the other side", short: "pts/hand", fmt: (v) => fmtSigned(v, 0) },
};
const MODE_HELP = {
  solo: {
    human_vs_system: "Play the deal yourself.",
    human_vs_agent: "You and the agent play the same deal side by side.",
    agent_vs_agent: "Watch the trained agent play.",
  },
  dealer: {
    human_vs_system: "Play against the dealer.",
    human_vs_agent: "You and the agent play the same shoe side by side.",
    agent_vs_agent: "Watch the trained agent play against the dealer.",
  },
  competitive: {
    human_vs_system: "You against the rule-based System bot.",
    human_vs_agent: "You against the trained agent.",
    agent_vs_agent: "Watch agents play. Swap any seat for a bot or random player.",
    agent_vs_counting: "The trained agent against the card-counting agent, in alternating seats.",
  },
};

const $ = (selector) => document.querySelector(selector);
const app = {
  games: [], game: null, mode: null, seats: [], session: null,
  paused: false, busy: false, timer: null, insight: null,
  autoAt: null, autoCancelled: false, seenCards: new Set(),
};

// ---------------------------------------------------------------------------
// Agent summaries
// ---------------------------------------------------------------------------
function agentTone(agent) {
  if (!agent?.trained) return "none";
  return agent.gate_passed ? "good" : agent.gate_passed === false ? "warn" : "none";
}

function metricText(agent) {
  const metric = METRICS[agent?.metric];
  return metric && agent.value != null ? `${metric.fmt(agent.value)} ${metric.short}` : null;
}

function targetText(agent) {
  const metric = METRICS[agent?.metric];
  if (agent?.max_gap != null) return `within ${agent.max_gap} of optimal`;
  if (!metric || agent?.target == null) return null;
  return `${agent.higher_is_better ? "≥" : "≤"} ${metric.fmt(agent.target)}`;
}

function agentChip(agent) {
  const tone = agentTone(agent);
  if (!agent?.trained) {
    return h("div", { class: "agent-chip none" }, h("i", { class: "status-dot" }),
      h("div", {}, h("strong", {}, "No trained agent"), h("span", {}, "The System bot stands in for the agent")));
  }
  const target = targetText(agent);
  return h("div", { class: `agent-chip ${tone}`, title: agent.algorithm_name ?? "" },
    h("i", { class: `status-dot ${tone}` }),
    h("div", {},
      h("strong", {}, `Agent · ${metricText(agent) ?? "trained"}`),
      h("span", {}, [target ? `target ${target}` : null, agent.gate_passed ? "met" : agent.gate_passed === false ? "not met" : null]
        .filter(Boolean).join(" · "))));
}

/** The card-counting agent's duel with the current agent, from the evaluator's report. */
function duelLine(agent) {
  const duel = agent.head_to_head;
  const metric = METRICS[duel?.metric];
  if (!duel || !metric) return null;
  const clear = Math.abs(duel.difference) > duel.ci95;
  const winner = duel.a_better ? "Card counting is ahead" : "The current agent is ahead";
  return h("p", { class: `duel-line ${clear ? (duel.a_better ? "good" : "warn") : ""}` },
    h("strong", {}, clear ? winner : "Head-to-head: too close to call"),
    ` · ${metric.fmt(duel.a)} vs ${metric.fmt(duel.b)} ${metric.short} over ${duel.deals.toLocaleString()} duplicate deals`);
}

function agentCard(agent, variant = "base") {
  const tone = agentTone(agent);
  const title = AGENT_TITLES[variant];
  if (!agent?.trained) {
    return h("div", { class: "intro-card agent-card" },
      h("h3", {}, title),
      h("p", { class: "muted" }, variant === "base"
        ? "No compatible trained model for this game yet. In agent seats the rule-based System bot plays instead."
        : "Not trained yet. It learns the same game with card-counting features added; until then the System bot takes its seats."));
  }
  const metric = METRICS[agent.metric];
  const rows = [
    ["Method", agent.algorithm_name],
    ["Measured on", agent.source === "test"
      ? `${agent.episodes?.toLocaleString() ?? "?"} held-out test deals`
      : `${agent.episodes?.toLocaleString() ?? "?"} validation deals`],
    ["Target", targetText(agent)],
    agent.baseline && metric ? [agent.baseline.label, metric.fmt(agent.baseline.value)] : null,
    agent.timesteps ? [agent.algorithm === "cfr" ? "Iterations" : "Training steps", Number(agent.timesteps).toLocaleString()] : null,
    agent.updated_at ? ["Updated", new Date(agent.updated_at).toLocaleString()] : null,
  ].filter((row) => row && row[1] != null);
  return h("div", { class: `intro-card agent-card ${tone}` },
    h("div", { class: "agent-card-head" },
      h("h3", {}, title),
      h("span", { class: `pill ${tone}` }, agent.gate_passed ? "Target met" : agent.gate_passed === false ? "Below target" : "Trained")),
    metric && agent.value != null ? h("div", { class: "big-metric" },
      h("span", { class: "value" }, metric.fmt(agent.value)),
      agent.ci95 != null ? h("span", { class: "ci" }, `± ${metric.fmt(agent.ci95).replace("+", "")}`) : null,
      h("span", { class: "unit" }, metric.label)) : null,
    h("dl", { class: "facts" }, rows.map(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
    duelLine(agent),
    agent.source === "validation"
      ? h("p", { class: "fine-print" }, "Validation score from training. Run the evaluator for the held-out test score.") : null);
}

// ---------------------------------------------------------------------------
// Setup: rail, modes, seats
// ---------------------------------------------------------------------------
function gameIcon(id, big = false) {
  const meta = GAME_META[id] ?? { icon: "?", hue: 150 };
  return h("span", { class: `game-icon ${big ? "big" : ""}`, style: `--hue:${meta.hue}`, "aria-hidden": "true" }, meta.icon);
}

function renderGameList() {
  $("#game-list").replaceChildren(...app.games.map((game) => {
    const tone = agentTone(game.agent);
    const selected = app.game?.id === game.id;
    return h("button", {
      type: "button",
      class: `game-item ${selected ? "selected" : ""}`,
      "aria-current": selected ? "true" : null,
      dataset: { game: game.id },
      onClick: () => selectGame(game.id),
    },
    gameIcon(game.id),
    h("span", { class: "game-text" },
      h("span", { class: "name" }, game.name),
      h("span", { class: "meta" }, GAME_META[game.id]?.players ?? `${game.players} players`)),
    h("i", { class: `status-dot ${tone}`, title: game.agent?.trained ? `Agent: ${metricText(game.agent) ?? "trained"}` : "No trained agent" }));
  }));
}

function defaultSeats(game, mode) {
  if (mode === "agent_vs_counting") return Array.from({ length: game.players }, (_, seat) => (seat % 2 ? "counting" : "agent"));
  const others = { human_vs_system: "system", human_vs_agent: "agent", agent_vs_agent: "agent" }[mode];
  const first = mode === "agent_vs_agent" ? "agent" : "human";
  return [first, ...Array(game.players - 1).fill(others)];
}

function renderSetup() {
  const game = app.game;
  $("#game-title").textContent = game.name;
  $("#game-desc").textContent = game.description;
  $("#agent-chip").replaceChildren(agentChip(game.agent));
  $("#mode-list").replaceChildren(...game.modes.map((mode) =>
    h("button", {
      type: "button",
      role: "radio",
      "aria-checked": app.mode === mode.id ? "true" : "false",
      class: `mode ${app.mode === mode.id ? "selected" : ""}`,
      title: MODE_HELP[game.category]?.[mode.id] ?? "",
      dataset: { mode: mode.id },
      onClick: () => selectMode(mode.id),
    }, mode.label)));
  const competitive = game.category === "competitive";
  $("#seat-section").hidden = !competitive;
  if (competitive) {
    $("#seat-config").replaceChildren(...app.seats.map((controller, seat) =>
      h("label", { class: `seat-row ctl-${controller}` },
        h("span", { class: "avatar sm", "aria-hidden": "true" }, AVATARS[controller]),
        h("span", { class: "sr-only" }, `Seat ${seat + 1}`),
        h("select", {
          dataset: { seat },
          "aria-label": `Seat ${seat + 1}`,
          onChange: (event) => { app.seats[seat] = event.target.value; renderSetup(); },
        }, game.controllers.map((c) => h("option", { value: c, selected: c === controller }, CONTROLLER_OPTIONS[c]))))));
  }
  $("#start").textContent = app.session ? "Restart" : "Start game";
}

function selectGame(id) {
  if (app.game?.id === id) return;
  app.game = app.games.find((g) => g.id === id);
  history.replaceState(null, "", `#${id}`);
  if (app.session && app.session.game !== id) endSession();
  renderGameList();
  if (!$("#rules").hidden) renderRules();
  selectMode(app.game.modes.some((m) => m.id === app.mode) ? app.mode : app.game.modes[0].id);
}

function selectMode(mode) {
  app.mode = mode;
  app.seats = defaultSeats(app.game, mode);
  renderSetup();
  if (!app.session) renderIntro();
}

function renderIntro() {
  const game = app.game;
  const mode = game.modes.find((m) => m.id === app.mode);
  $("#boards").className = "boards";
  $("#boards").replaceChildren(h("div", { class: "intro" },
    h("div", { class: "intro-card" },
      h("div", { class: "intro-hero" }, gameIcon(game.id, true),
        h("div", {}, h("h2", {}, game.name), h("p", { class: "muted" }, game.description))),
      h("h3", {}, "How to play"),
      h("ol", { class: "guide" }, (GUIDES[game.id] ?? []).map((line) => h("li", {}, line))),
      h("button", { class: "link-btn", type: "button", id: "intro-rules", onClick: () => openRules() },
        "Read the full rules, controls and tips →"),
      h("div", { class: "intro-cta" },
        h("button", { class: "btn btn-primary btn-lg", type: "button", id: "intro-start", onClick: start }, `Start · ${mode.label}`),
        h("span", { class: "muted" }, MODE_HELP[game.category]?.[app.mode] ?? ""))),
    h("div", { class: "agent-cards" },
      Object.entries(game.agents ?? { base: game.agent }).map(([variant, status]) => agentCard(status, variant)))));
  document.body.dataset.phase = "idle";
  document.body.classList.remove("in-session");
  setStatus("Ready");
  $("#round-bar").hidden = true;
  renderInsight();
  renderStats();
  $("#log").replaceChildren();
}

// ---------------------------------------------------------------------------
// Session lifecycle and the computer-move loop
// ---------------------------------------------------------------------------
function endSession() {
  stopTimer();
  if (app.session) api.remove(app.session.id).catch(() => {});
  app.session = null;
  app.insight = null;
  app.seenCards = new Set();
}

async function start() {
  if (app.game.category === "competitive" && app.seats.filter((s) => s === "human").length > 1) {
    return toast("Only one seat can be you.");
  }
  endSession();
  const body = { game: app.game.id, mode: app.mode };
  if (app.game.category === "competitive") body.seats = [app.seats.slice()];
  const seed = $("#seed").value;
  if (seed !== "") body.seed = Number(seed);
  setStatus(`Dealing ${app.game.name}…`);
  try {
    app.session = await api.create(body);
  } catch (error) {
    toast(error.message);
    return renderIntro();
  }
  app.autoCancelled = false;
  renderSetup();
  render();
  schedule();
}

function stopTimer() {
  clearTimeout(app.timer);
  app.timer = null;
  app.autoAt = null;
}

function speed() {
  return Number($("#speed").value);
}

function pendingBoards() {
  return app.session ? app.session.boards.flatMap((b, i) => (!b.done && !b.human_turn ? [i] : [])) : [];
}

function canAutoNext() {
  const session = app.session;
  return session && session.boards.every((b) => b.done) && $("#auto-next").checked && !app.paused
    && !app.autoCancelled && session.boards.every((b) => b.human_seat == null);
}

function schedule() {
  stopTimer();
  const session = app.session;
  if (!session || app.busy) return renderCountdown();
  const pending = pendingBoards();
  if (pending.length) {
    // A renderer may ask for a pause so a finished trick can be read (shorter when fast, none at Instant).
    const hold = Math.max(0, ...pending.map((i) => RENDERERS[session.game].holdAfter?.(session.boards[i]) ?? 0));
    const scale = [1, 1, 1, 0.6, 0][speed()];
    if (!app.paused) app.timer = setTimeout(tick, SPEEDS[speed()] + hold * scale);
  } else if (canAutoNext()) {
    app.autoAt = Date.now() + AUTO_NEXT_MS;
    app.timer = setTimeout(nextRound, AUTO_NEXT_MS);
  }
  renderCountdown();
}

function captureInsight(state) {
  const moves = state.advanced ?? [];
  const move = [...moves].reverse().find((m) => m.controller !== "human");
  if (move) app.insight = { ...move, who: seatName(state, move.board, move.seat) };
}

/**
 * Run ``call(sessionId)`` for the current session with the busy flag held, then redraw.
 * A reply that arrives after the session ended (game switched or restarted) is dropped.
 */
async function sessionCall(call, onError = () => {}) {
  const id = app.session?.id;
  if (!id) return;
  stopTimer();
  app.busy = true;
  let stale = false;
  try {
    stale = (await call(id)) === false;
  } catch (error) {
    stale = app.session?.id !== id;
    if (!stale) { toast(error.message); onError(); }
  } finally {
    app.busy = false;
  }
  if (!stale) render();
  schedule();
}

/** Store a state returned for session ``id``; false if that session is no longer current. */
function accept(id, state) {
  if (app.session?.id !== id) return false;
  app.session = state;
  return true;
}

async function tick() {
  const pending = pendingBoards();
  if (!pending.length || app.busy) return schedule();
  const steps = speed() === SPEEDS.length - 1 ? 25 : 1;
  await sessionCall(async (id) => {
    for (const board of pending) {
      const state = await api.advance(id, { board, steps });
      if (!accept(id, state)) return false;
      captureInsight(state);
    }
  }, () => setPaused(true));
}

async function act(board, payload) {
  if (app.busy) return void setTimeout(() => act(board, payload), 60);
  await sessionCall(async (id) => accept(id, await api.action(id, { board, ...payload })));
}

async function boardCall(kind, board, busyText) {
  if (app.busy) return;
  if (busyText) setStatus(busyText);
  await sessionCall(async (id) => {
    const result = await api[kind](id, { board });
    return result.boards ? accept(id, result) : app.session?.id === id;
  });
}

async function nextRound() {
  if (!app.session || app.busy || !app.session.boards.every((b) => b.done)) return;
  await sessionCall(async (id) => {
    if (!accept(id, await api.newRound(id))) return false;
    app.autoCancelled = false;
  });
}

function setPaused(paused) {
  app.paused = paused;
  const button = $("#pause");
  button.setAttribute("aria-pressed", String(paused));
  button.querySelector(".label").textContent = paused ? "Resume" : "Pause";
  button.querySelector(".ico").innerHTML = paused ? ICONS.play : ICONS.pause;
  if (app.session) renderStatus();
  schedule();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function seatName(session, boardIndex, seat) {
  const board = session.boards[boardIndex];
  const controller = board.seats[seat];
  const base = CONTROLLER_NAMES[controller];
  const same = board.seats.filter((c) => c === controller).length;
  const name = same > 1 ? `${base} ${seat + 1}` : base;
  return board.fallback[seat] ? `${name} (System)` : name;
}

function boardTitle(session, index) {
  if (!session.duel) return null;
  return index === 0 ? "You" : "Agent (same deal)";
}

/** Cards that appeared in a new place since the last frame get a short deal-in animation. */
function animateNewCards(root) {
  const seen = new Set();
  let fresh = 0;
  root.querySelectorAll("[data-card]").forEach((el) => {
    const board = el.closest("[data-board]")?.dataset.board ?? "";
    const key = `${board}|${el.parentElement?.classList[0] ?? ""}|${el.dataset.card}`;
    seen.add(key);
    if (!app.seenCards.has(key)) {
      el.classList.add("enter");
      el.style.setProperty("--delay", `${Math.min(fresh++, 14) * 18}ms`);
    }
  });
  app.seenCards = seen;
}

function render() {
  const session = app.session;
  if (!session) return renderIntro();
  const renderer = RENDERERS[session.game];
  const boardsEl = $("#boards");
  boardsEl.className = `boards ${session.duel ? "duel" : ""}`;
  boardsEl.replaceChildren(...session.boards.map((board, index) => {
    const root = h("div", { class: "board-body" });
    renderer.render(root, {
      session, board, index,
      seatName: (seat) => seatName(session, index, seat),
      act: (action) => act(index, { action }),
      actCards: (cards) => act(index, { cards }),
      actClaim: (cards, rank) => act(index, { cards, claim_rank: rank }),
      undo: () => boardCall("undo", index),
      solve: () => boardCall("solve", index, "Solving the deal with every card visible…"),
      rerender: render,
    });
    const title = boardTitle(session, index);
    const controller = index === 0 ? "human" : "agent";
    return h("article", { class: `board ${board.human_turn ? "your-turn" : ""} ${board.done ? "finished" : ""}`, dataset: { board: index } },
      title ? h("header", { class: `board-head ctl-${controller}` },
        h("span", { class: "avatar sm", "aria-hidden": "true" }, AVATARS[controller]), h("h3", { class: "board-title" }, title),
        board.done ? h("span", { class: "pill" }, "Finished") : board.human_turn ? h("span", { class: "pill accent" }, "Your move") : null) : null,
      root);
  }));
  animateNewCards(boardsEl);
  document.body.classList.add("in-session");
  renderStatus();
  renderRoundBar();
  renderInsight();
  renderStats();
  renderLog();
}

function renderStatus() {
  const session = app.session;
  const allDone = session.boards.every((b) => b.done);
  const humanTurn = session.boards.some((b) => b.human_turn);
  let phase = "bots";
  let text = app.paused ? "Paused" : session.boards.some((b) => b.seats.includes("agent") || b.seats.includes("counting")) ? "Agent is thinking…" : "Computer is playing…";
  if (allDone) { phase = "done"; text = session.match_over ? "Match over" : "Deal finished"; }
  else if (humanTurn) { phase = "human"; text = "Your turn"; }
  document.body.dataset.phase = phase;
  document.body.classList.toggle("paused", app.paused);
  setStatus(`${text} · deal ${session.round + 1}`);
}

function boardOutcome(session, index) {
  const board = session.boards[index];
  if (!board.done) return { text: null, tone: "" };
  const view = board.view;
  if (board.seats.length === 1) {
    if ("won" in view) return { text: view.won ? "solved the deal" : `${view.foundation_count}/52 cards home`, tone: view.won ? "win" : "loss" };
    return { text: `${fmtSigned(view.net)} units`, tone: view.net > 0 ? "win" : view.net < 0 ? "loss" : "" };
  }
  if (board.truncated) return { text: "stopped at the move limit", tone: "" };
  const names = board.winners.map((seat) => seatName(session, index, seat));
  const humanWon = board.human_seat != null && board.winners.includes(board.human_seat);
  const tone = board.human_seat == null ? "" : humanWon ? "win" : "loss";
  if (!names.length) return { text: "No winner this deal", tone: "" };
  return { text: `${names.join(" & ")} ${names.length > 1 || names[0] === "You" ? "win" : "wins"}`, tone };
}

function renderRoundBar() {
  const session = app.session;
  const done = session.boards.every((b) => b.done);
  const bar = $("#round-bar");
  bar.hidden = !done;
  if (!done) return;
  const outcomes = session.boards.map((_, i) => boardOutcome(session, i));
  const text = outcomes.map((o, i) => {
    const title = boardTitle(session, i);
    return title ? `${title}: ${o.text}` : o.text;
  }).join("  ·  ");
  const tones = outcomes.map((o) => o.tone);
  const tone = session.duel ? "" : tones[0];
  bar.className = `result-banner ${tone}`;
  $("#round-eyebrow").textContent = session.match_over ? "Match over" : `Deal ${session.round + 1} finished`;
  $("#round-result").textContent = text;
  $("#new-round").textContent = session.match_over ? "New match" : "Next deal";
  renderCountdown();
}

function renderCountdown() {
  const el = $("#auto-countdown");
  const cancel = $("#cancel-auto");
  if (!el) return;
  if (app.autoAt) {
    const left = Math.max(0, Math.ceil((app.autoAt - Date.now()) / 1000));
    el.textContent = `Next deal in ${left} s`;
    cancel.hidden = false;
  } else {
    el.textContent = app.session?.boards.every((b) => b.done) && !$("#auto-next").checked ? "Press N or the button when you're ready." : "";
    cancel.hidden = true;
  }
}

function renderInsight() {
  const move = app.insight;
  const el = $("#insight");
  if (!move || !app.session) {
    el.replaceChildren(h("p", { class: "empty-note" }, "When a computer player moves, its choice and the agent's top alternatives appear here."));
    return;
  }
  const variant = VARIANT_OF[move.controller];
  const agent = (variant && app.session.agents?.[variant]) || app.session.agent;
  const game = app.games.find((g) => g.id === app.session.game);
  const trained = Boolean(variant) && !move.fallback;
  const source = trained ? `${agent.algorithm_name ?? "Trained agent"}${variant === "counting" ? " + card counting" : ""}`
    : variant ? "No trained agent: System rule"
    : move.controller === "random" ? "Random legal move" : "Rule-based System bot";
  const top = move.top ?? [];
  const bars = top.map((t) =>
    h("div", { class: `bar-row ${t.action === move.action ? "chosen" : ""}` },
      h("span", { class: "bar-label", title: t.label }, t.label),
      h("span", { class: "bar", role: "meter", "aria-valuemin": 0, "aria-valuemax": 100, "aria-valuenow": Math.round(t.prob * 100), "aria-label": t.label },
        h("span", { class: "fill", style: `width:${Math.max(2, Math.round(t.prob * 100))}%` })),
      h("span", { class: "bar-value" }, `${(t.prob * 100).toFixed(1)}%`)));
  let note = null;
  if (move.rule_move) note = "This situation was outside the agent's trained strategy, so it used the System rule.";
  else if (trained && agent.algorithm === "search") note = "Look-ahead search computes one best move, so it reports only its choice.";
  else if (trained && top.length === 1 && !app.session.reveal && game?.category === "competitive") note = "Alternatives stay hidden while you play: they could reveal the agent's cards.";
  else if (!top.length) note = "Rule-based moves have no probabilities.";
  const controller = move.controller;
  fill(el,
    h("div", { class: `insight-head ctl-${controller}` },
      h("span", { class: "avatar", "aria-hidden": "true" }, AVATARS[controller]),
      h("div", {}, h("strong", {}, move.who), h("span", { class: "muted" }, source))),
    h("p", { class: "insight-choice" }, "Chose ", h("strong", {}, move.label)),
    bars.length ? h("div", { class: "bars" }, bars) : null,
    note ? h("p", { class: "fine-print" }, note) : null,
  );
}

function renderStats() {
  const session = app.session;
  const el = $("#stats");
  if (!session) return el.replaceChildren(h("p", { class: "empty-note" }, "Results are tallied here once a game starts."));
  const extra = { blackjack: ["Net units", (s, i) => fmtSigned(s.totals[i])],
                  poker: ["Chips", (s, i) => fmtSigned(s.points[i], 0)],
                  hearts: [`Points (to ${session.match_target})`, (s, i) => s.points[i]],
                  donkey: ["Times Donkey", (s, i) => +s.points[i].toFixed(2)],
                  canasta: [`Match (to ${session.match_target})`, (s, i) => s.points[i]] }[session.game];
  el.replaceChildren(...session.boards.map((board, index) => {
    const s = board.stats;
    const title = boardTitle(session, index);
    const best = Math.max(...s.wins);
    return h("div", { class: "stats-board" },
      h("p", { class: "stats-caption" }, title ? h("strong", {}, `${title} · `) : null, `${s.rounds} deal${s.rounds === 1 ? "" : "s"} finished`),
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Player"), h("th", { scope: "col", class: "num" }, "Wins"),
          extra ? h("th", { scope: "col", class: "num" }, extra[0]) : null)),
        h("tbody", {}, board.seats.map((controller, seat) =>
          h("tr", { class: s.rounds && s.wins[seat] === best && best > 0 ? "leader" : "" },
            h("td", {}, h("span", { class: `who ctl-${controller}` }, h("span", { class: "avatar xs", "aria-hidden": "true" }, AVATARS[controller]),
              seatName(session, index, seat))),
            h("td", { class: "num" }, s.wins[seat]),
            extra ? h("td", { class: "num" }, extra[1](s, seat)) : null)))));
  }));
}

function renderLog() {
  const session = app.session;
  const items = [];
  session.boards.forEach((board, index) => {
    // Duel boards have one seat each, so the seat name ("You", "Agent") already says which board.
    board.moves.forEach((m, order) => items.push({ order, index, controller: m.controller,
      who: seatName(session, index, m.seat), what: m.label }));
  });
  items.sort((a, b) => b.order - a.order || a.index - b.index);
  $("#log").replaceChildren(...items.slice(0, 40).map((item) =>
    h("li", { class: `ctl-${item.controller}` }, h("span", { class: "who" }, item.who), h("span", { class: "what" }, item.what))));
}

/** replaceChildren that skips null/false (replaceChildren would print them as text). */
function fill(el, ...children) {
  el.replaceChildren(...children.filter((c) => c != null && c !== false));
}

function setStatus(text) {
  $("#status .status-text").textContent = text;
}

function toast(message, tone = "error") {
  const el = h("div", { class: `toast ${tone}`, role: tone === "error" ? "alert" : "status" },
    h("span", {}, message),
    h("button", { class: "icon-btn", type: "button", "aria-label": "Dismiss", onClick: () => el.remove() }, "✕"));
  $("#toasts").append(el);
  setTimeout(() => el.remove(), 5000);
}

// ---------------------------------------------------------------------------
// Help dialog and controls
// ---------------------------------------------------------------------------
// ---- How to play: a side panel with one collapsible section per topic --------
function ruleSection(section, open) {
  const body = [];
  if (section.text) body.push(h("p", {}, section.text));
  if (section.items) body.push(h("ul", { class: "rule-list" }, section.items.map((item) => h("li", {}, item))));
  if (section.table) body.push(h("dl", { class: "rule-table" }, section.table.map(([term, meaning]) => [h("dt", {}, term), h("dd", {}, meaning)])));
  return h("details", { class: "rule-section", open },
    h("summary", {}, h("span", {}, section.title), h("span", { class: "chevron", "aria-hidden": "true" }, "›")),
    h("div", { class: "rule-body" }, body));
}

function renderRules() {
  const game = app.game;
  $("#rules-title").textContent = game.name;
  const sections = RULES[game.id] ?? [];
  fill($("#rules-body"),
    h("div", { class: "rules-summary" }, gameIcon(game.id), h("p", { class: "muted" }, game.description)),
    ...sections.map((section, i) => ruleSection(section, i < 2)),
    h("p", { class: "fine-print" }, "The complete rules, including edge cases, are in docs/GAMES.md."));
  syncRulesToggle();
}

function syncRulesToggle() {
  const all = [...document.querySelectorAll("#rules-body .rule-section")];
  $("#rules-toggle-all").textContent = all.length && all.every((d) => d.open) ? "Collapse all" : "Expand all";
}

function openRules(toggle = false) {
  const drawer = $("#rules");
  if (toggle && !drawer.hidden) return closeRules();
  renderRules();
  drawer.hidden = false;
  document.body.classList.add("rules-open");
  $("#rules-open").setAttribute("aria-expanded", "true");
  $("#rules-title").focus({ preventScroll: true });
}

function closeRules() {
  $("#rules").hidden = true;
  document.body.classList.remove("rules-open");
  $("#rules-open").setAttribute("aria-expanded", "false");
}

function openHelp() {
  const game = app.game;
  $("#help-title").textContent = "Help";
  fill($("#help-body"),
    game ? h("button", { class: "btn", type: "button", onClick: () => { $("#help").close(); openRules(); } },
      `How to play ${game.name}`) : null,
    h("h3", {}, "Players"),
    h("ul", { class: "legend" }, Object.entries(CONTROLLER_OPTIONS).map(([c, label]) =>
      h("li", { class: `ctl-${c}` }, h("span", { class: "avatar xs", "aria-hidden": "true" }, AVATARS[c]), label))),
    h("h3", {}, "Keyboard shortcuts"),
    h("table", { class: "shortcuts" }, h("tbody", {}, SHORTCUTS.map(([key, what]) =>
      h("tr", {}, h("td", {}, h("kbd", {}, key)), h("td", {}, what))))),
    h("p", { class: "fine-print" }, "Full rules for every game are in docs/GAMES.md."));
  $("#help").showModal();
}

function wireControls() {
  $("#start").addEventListener("click", start);
  $("#new-round").addEventListener("click", nextRound);
  $("#cancel-auto").addEventListener("click", () => { app.autoCancelled = true; schedule(); });
  $("#pause").addEventListener("click", () => setPaused(!app.paused));
  $("#step").addEventListener("click", () => { stopTimer(); tick(); });
  const speedLabel = () => { $("#speed-label").textContent = SPEED_NAMES[speed()]; };
  $("#speed").addEventListener("input", () => { speedLabel(); schedule(); });
  $("#auto-next").addEventListener("change", () => { app.autoCancelled = false; schedule(); });
  $("#help-open").addEventListener("click", openHelp);
  $("#rules-open").addEventListener("click", () => openRules(true));
  $("#rules-close").addEventListener("click", () => { closeRules(); $("#rules-open").focus(); });
  $("#rules-toggle-all").addEventListener("click", () => {
    const all = [...document.querySelectorAll("#rules-body .rule-section")];
    const open = !all.every((d) => d.open);
    all.forEach((d) => { d.open = open; });
    syncRulesToggle();
  });
  $("#rules-body").addEventListener("toggle", syncRulesToggle, true);
  $("#help").addEventListener("click", (event) => {
    if (event.target === event.currentTarget || event.target.closest("[data-close]")) $("#help").close();
  });
  setInterval(() => app.autoAt && renderCountdown(), 250);
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey || event.altKey || $("#help").open) return;
    if (event.key === "Escape" && !$("#rules").hidden) return closeRules();
    if (["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) return;
    const key = event.key.toLowerCase();
    if (key === " " && event.target.tagName !== "BUTTON") { event.preventDefault(); setPaused(!app.paused); }
    else if (key === "s") { stopTimer(); tick(); }
    else if (key === "n") nextRound();
    else if (key === "r") start();
    else if (key === "h") openRules(true);
    else if (key === "?") openHelp();
  });
  speedLabel();
}

async function init() {
  wireControls();
  try {
    app.games = await api.games();
  } catch (error) {
    setStatus("Backend unavailable");
    toast(`Backend unavailable: ${error.message}`);
    return;
  }
  const fromHash = () => location.hash.slice(1);
  selectGame(app.games.some((g) => g.id === fromHash()) ? fromHash() : app.games[0].id);
  window.addEventListener("hashchange", () => {
    if (app.games.some((g) => g.id === fromHash()) && fromHash() !== app.game?.id) selectGame(fromHash());
  });
}

init();
