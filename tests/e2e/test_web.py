"""Browser end-to-end tests: every game in every mode through the real web UI.

Uses the installed Microsoft Edge (or Playwright's Chromium if downloaded), and a
stand-in "agent" so the tests don't depend on locally trained model files.
"""
from __future__ import annotations

import socket
import threading
import time

import pytest
import uvicorn

from card_games_rl.api.app import create_app
from card_games_rl.api.sessions import AgentCache
from card_games_rl.core import Policy
from card_games_rl.games import GAME_SPECS, heuristic_policy

pytestmark = pytest.mark.e2e
playwright_api = pytest.importorskip("playwright.sync_api")
GAMES = sorted(GAME_SPECS)


class UniformAgent(Policy):
    """Plays like the System bot but reports probabilities, like a model would."""

    def __init__(self, game_id):
        self.inner = heuristic_policy(game_id, seed=0)

    def act(self, game, player):
        return self.inner.act(game, player)

    def action_probabilities(self, game, player):
        legal = game.legal_actions()
        return {a: 1 / len(legal) for a in legal}


@pytest.fixture(scope="module")
def base_url():
    app = create_app(agents=AgentCache(loader=lambda game_id, root=None: UniformAgent(game_id)))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with playwright_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:
            try:
                browser = p.chromium.launch()
            except Exception as exc:  # pragma: no cover - depends on the machine
                pytest.skip(f"No browser available for Playwright: {exc}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser, base_url):
    page = browser.new_page(viewport={"width": 1500, "height": 1000})
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.base_url = base_url
    yield page
    page.close()
    assert not errors, f"JavaScript errors: {errors}"


def start(page, game, mode, instant=False, auto_next=False):
    page.goto(f"{page.base_url}/#{game}")
    page.wait_for_selector(f"[data-game={game}].selected")
    page.click(f"[data-mode={mode}]")
    if instant:
        page.eval_on_selector("#speed", "el => { el.value = 4; el.dispatchEvent(new Event('input')); }")
    if not auto_next:
        page.uncheck("#auto-next")
    page.click("#start")
    page.wait_for_selector("body[data-phase=human], body[data-phase=done], body[data-phase=bots]")


def human_move(page):
    """Make one human move through the UI and check it was applied."""
    page.wait_for_selector("body[data-phase=human]", timeout=20000)
    before = page.locator("#log li").count()
    play = page.locator("[id^=play-cards]:visible")
    if play.count():  # Bluff: pick a card, then claim
        page.locator(".board.your-turn .card.clickable:visible").first.click()
        play.first.click()
    else:
        page.locator(".board.your-turn [data-action]:visible:not([disabled])").first.click()
    page.wait_for_function(f"document.querySelectorAll('#log li').length > {before}")


@pytest.mark.parametrize("game", GAMES)
def test_human_vs_system_move(page, game):
    start(page, game, "human_vs_system")
    human_move(page)
    assert page.locator(".board").count() == 1


@pytest.mark.parametrize("game", GAMES)
def test_human_vs_agent(page, game):
    start(page, game, "human_vs_agent", instant=True)
    duel = GAME_SPECS[game].category in ("solo", "dealer")
    assert page.locator(".board").count() == (2 if duel else 1)
    human_move(page)
    if not duel:  # the agents answer and their choices show up in the insight panel
        # Hearts: the human passes three cards before anyone else moves.
        # Poker: folding first ends the hand before the agent acts, so deal again.
        for _ in range(8):
            page.wait_for_selector("body[data-phase=human], body[data-phase=done]", timeout=20000)
            if page.locator("#insight .bar-row").count():
                break
            if page.get_attribute("body", "data-phase") == "done":
                page.click("#new-round")
                page.wait_for_selector("body:not([data-phase=done])")
                continue
            human_move(page)
        page.wait_for_selector("#insight .bar-row", timeout=20000)


@pytest.mark.parametrize("game", GAMES)
def test_agent_vs_agent_runs_to_the_end(page, game):
    start(page, game, "agent_vs_agent", instant=True)
    page.wait_for_selector("body[data-phase=done]", timeout=90000)
    assert page.locator("#round-bar").is_visible()
    assert "1 deal finished" in page.locator("#stats").inner_text()
    page.wait_for_selector("#insight .bar-row")
    page.click("#new-round")
    page.wait_for_function("document.querySelector('#stats').innerText.includes('1 deal finished')")


def test_klondike_click_to_move(page):
    """Pick a source then a destination, the way a person plays."""
    start(page, "klondike", "human_vs_system")
    for _ in range(40):  # draw until some non-stock move exists
        moves = page.locator(".legal-moves [data-action]")
        page.locator(".legal-moves summary").click()
        labels = [moves.nth(i).inner_text() for i in range(moves.count())]
        if any(not label.startswith(("Draw", "Recycle")) for label in labels):
            break
        page.locator(".stock [data-action]").first.click()
        page.wait_for_selector("body[data-phase=human]")
    before = page.locator("#log li").count()
    page.locator(".legal-moves [data-action]").filter(has_not_text="Draw").filter(has_not_text="Recycle").first.click()
    page.wait_for_function(f"document.querySelectorAll('#log li').length > {before}")


def test_hearts_custom_seats_and_pause(page):
    page.goto(f"{page.base_url}/#hearts")
    page.click("[data-mode=agent_vs_agent]")
    page.select_option("[data-seat='1']", "random")
    page.select_option("[data-seat='2']", "system")
    page.click("#pause")
    page.click("#start")
    page.wait_for_selector("body[data-phase=bots]")
    page.click("#step")
    page.wait_for_function("document.querySelectorAll('#log li').length === 1")
    time.sleep(0.6)
    assert page.locator("#log li").count() == 1  # paused: nothing else moved
    assert "Random 2" not in page.locator("#boards").inner_text()
    assert "Random" in page.locator("#boards").inner_text()


@pytest.mark.parametrize("game", GAMES)
def test_how_to_play_panel_opens_for_every_game(page, game):
    page.goto(f"{page.base_url}/#{game}")
    page.wait_for_selector(f"[data-game={game}].selected")
    page.click("#rules-open")
    page.wait_for_selector("#rules:not([hidden]) .rule-section")
    assert page.locator("#rules .rule-section").count() >= 4
    assert page.locator("#rules-title").inner_text() == page.locator("#game-title").inner_text()
    page.click("#rules-toggle-all")
    assert page.locator("#rules .rule-section[open]").count() == page.locator("#rules .rule-section").count()
    page.click("#rules-close")
    assert page.locator("#rules").is_hidden()


def test_how_to_play_stays_open_while_playing(page):
    start(page, "hearts", "human_vs_system")
    page.keyboard.press("h")
    page.wait_for_selector("#rules:not([hidden])")
    human_move(page)  # the table is still usable beside the panel
    page.keyboard.press("Escape")
    assert page.locator("#rules").is_hidden()
