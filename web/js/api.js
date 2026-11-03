// Thin client for the FastAPI backend (card_games_rl/api/app.py).

async function request(method, path, body) {
  const response = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail ?? response.statusText);
    throw new Error(detail);
  }
  return data;
}

export const api = {
  games: () => request("GET", "/api/games"),
  create: (body) => request("POST", "/api/sessions", body),
  get: (id) => request("GET", `/api/sessions/${id}`),
  action: (id, body) => request("POST", `/api/sessions/${id}/action`, body),
  advance: (id, body) => request("POST", `/api/sessions/${id}/advance`, body),
  newRound: (id) => request("POST", `/api/sessions/${id}/new-round`),
  undo: (id, body) => request("POST", `/api/sessions/${id}/undo`, body),
  solve: (id, body) => request("POST", `/api/sessions/${id}/solve`, body),
  remove: (id) => request("DELETE", `/api/sessions/${id}`),
};
