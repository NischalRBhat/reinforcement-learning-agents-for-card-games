// One renderer per game: render(root, ctx) draws the board and wires human input.
import * as blackjack from "./blackjack.js";
import * as bluff from "./bluff.js";
import * as donkey from "./donkey.js";
import * as hearts from "./hearts.js";
import * as klondike from "./klondike.js";
import * as poker from "./poker.js";
import * as rummy from "./rummy.js";

export const RENDERERS = { blackjack, bluff, donkey, hearts, klondike, poker, rummy };
