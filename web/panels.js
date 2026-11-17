// The galaxy's two lists: "For you" (content-based recommendations, GET /api/recommendations)
// and Ask Sonata (RAG answers, POST /api/ask) shown as cards. All text from songs and from the
// model is set with textContent, never as HTML.

const SOURCE_LABELS = {
  gemini: "Answered by Gemini",
  cache: "Answered from cache (no API call)",
  mock: "Mock mode: reasons are canned, songs are really retrieved",
  fallback: "Retrieval only",
};

const ASK_TIMEOUT_MS = 90_000;
// What to say while a model call is running, by elapsed time
const PROGRESS = [
  [0, "Searching your library"],
  [3000, "Still working (the first question after starting Sonata also loads the search model)"],
  [12000, "Waiting for the model to answer"],
];

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function skeleton(width) {
  const line = el("span", "skeleton");
  line.style.width = width;
  return line;
}

// Children fade up one after another
function stagger(list, items) {
  items.forEach((item, i) => item.style.setProperty("--i", i));
  list.classList.add("stagger");
  list.replaceChildren(...items);
}

// Hover or keyboard focus highlights the song's star; leaving clears it
function linkToStar(button, id, onHighlight) {
  button.addEventListener("mouseenter", () => onHighlight(id));
  button.addEventListener("focus", () => onHighlight(id));
  button.addEventListener("mouseleave", () => onHighlight(null));
  button.addEventListener("blur", () => onHighlight(null));
}

// ---------- For you ----------

// Placeholder rows; shown from page load until the recommendations arrive
export function showRecommendationsLoading(list) {
  list.setAttribute("aria-busy", "true");
  list.classList.remove("stagger");
  list.replaceChildren(...Array.from({ length: 5 }, () => {
    const item = el("li", "rec-skeleton");
    item.append(skeleton("60%"), skeleton("40%"), skeleton("75%"));
    return item;
  }));
}

export async function loadRecommendations({ list, findTrack, onLoaded, onOpen, onHighlight }) {
  showRecommendationsLoading(list);

  let recs;
  try {
    const response = await fetch("/api/recommendations");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    recs = await response.json();
  } catch (error) {
    console.warn("Could not load recommendations", error);
    list.replaceChildren(el("li", "panel-sub", "Couldn't load recommendations."));
    return;
  } finally {
    list.setAttribute("aria-busy", "false");
  }
  if (!recs.length) {
    list.replaceChildren(el("li", "panel-sub", "No recommendations: like some songs first."));
    return;
  }
  onLoaded(recs);

  stagger(list, recs.map((rec, i) => {
    const track = findTrack(rec.deezer_id);
    const button = el("button", "rec");
    button.type = "button";
    const top = el("span", "rec-top");
    top.append(el("span", "rec-rank", String(i + 1).padStart(2, "0")), el("span", "rec-title", rec.title));
    const why = el("span", "rec-why", `Like ${rec.closest_liked.title} · ${rec.similarity.toFixed(2)}`);
    why.title = `Closest liked song: ${rec.closest_liked.title} by ${rec.closest_liked.artist}`;
    button.append(top, el("span", "rec-meta", [rec.artist, track?.genre].filter(Boolean).join(" · ")), why);
    for (const shared of rec.shared_features) {
      // "brightness: both brighter than average (+1.3σ, +1.1σ)" -> keep the words, full text on hover
      const line = el("span", "rec-shared", shared.replace(/\s*\([^()]*σ[^()]*\)$/, ""));
      line.title = shared;
      button.append(line);
    }
    linkToStar(button, rec.deezer_id, onHighlight);
    button.addEventListener("click", () => track && onOpen(track));
    const item = el("li");
    item.append(button);
    return item;
  }));
}

// ---------- Ask Sonata ----------

export function initAsk({ form, input, button, answers, status, cards, clear, findTrack, onAnswers, onOpen, onHighlight }) {
  let request = null;           // {controller, timers} of the call in flight

  const showStatus = (text, { error = false, working = false } = {}) => {
    status.textContent = text;
    status.classList.toggle("is-error", error);
    status.classList.toggle("is-working", working);
  };

  const setBusy = (busy) => {
    form.classList.toggle("is-busy", busy);
    button.disabled = busy;
    button.querySelector(".label").textContent = busy ? "Asking" : "Ask";
    answers.setAttribute("aria-busy", String(busy));
  };

  const cancel = () => {
    if (!request) return;
    request.controller.abort();
    request.timers.forEach(clearTimeout);
    request = null;
    setBusy(false);
  };

  const reset = () => {
    cancel();
    answers.hidden = true;
    cards.replaceChildren();
    showStatus("");
    onHighlight(null);
    onAnswers([]);
  };

  clear.addEventListener("click", () => {
    reset();
    input.focus();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    cancel();                                   // a new question replaces one still running
    const controller = new AbortController();
    const timers = PROGRESS.map(([after, text]) => setTimeout(() => showStatus(text, { working: true }), after));
    timers.push(setTimeout(() => controller.abort("timeout"), ASK_TIMEOUT_MS));
    const current = (request = { controller, timers });

    setBusy(true);
    answers.hidden = false;
    onHighlight(null);
    onAnswers([]);
    cards.classList.remove("stagger");
    cards.replaceChildren(...Array.from({ length: 3 }, () => {
      const card = el("div", "card is-skeleton");
      card.setAttribute("aria-hidden", "true");
      card.append(skeleton("30%"), skeleton("70%"), skeleton("45%"), skeleton("90%"), skeleton("80%"));
      const item = el("li");
      item.append(card);
      return item;
    }));

    try {
      // Empty or too-long requests are still sent: sonata.rag owns those rules and explains them
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: input.value }),
        signal: controller.signal,
      });
      const body = await response.json().catch(() => ({}));
      if (current !== request) return;          // superseded by a newer question
      if (!response.ok) {
        cards.replaceChildren();
        const detail = typeof body.detail === "string" ? body.detail : `Sonata couldn't answer (HTTP ${response.status}).`;
        showStatus(detail, { error: true });
        return;
      }
      render(body);
    } catch (error) {
      if (current !== request) return;
      cards.replaceChildren();
      const timedOut = controller.signal.reason === "timeout";
      showStatus(timedOut ? "Sonata took too long to answer. Try again in a moment."
                          : "Couldn't reach Sonata's server. Is python -m sonata.server running?", { error: true });
      if (!timedOut) console.warn("Ask failed", error);
    } finally {
      if (current === request) {
        timers.forEach(clearTimeout);
        request = null;
        setBusy(false);
      }
    }
  });

  function render(body) {
    const parts = [SOURCE_LABELS[body.source] || body.source];
    if (body.message) parts.push(body.message);
    if (body.latency_ms != null) parts.push(`${(body.latency_ms / 1000).toFixed(1)} s`);
    showStatus(parts.join(" · "), { error: body.source === "fallback" });

    stagger(cards, body.recommendations.map((pick, i) => {
      const track = findTrack(pick.deezer_id);
      const card = el("button", "card");
      card.type = "button";
      card.style.setProperty("--i", i);
      const confidence = Math.max(0, Math.min(1, pick.confidence));
      const top = el("span", "card-top");
      const conf = el("span", "conf");
      const bar = el("span", "conf-bar");
      const fill = el("span");
      fill.style.width = `${Math.round(confidence * 100)}%`;
      bar.append(fill);
      conf.append(bar, el("span", null, confidence.toFixed(2)));
      conf.title = "Confidence: how closely this song matched your request in retrieval (0-1)";
      top.append(el("span", "card-rank", String(i + 1)), conf);
      card.append(
        top,
        el("span", "card-title", pick.title),
        el("span", "card-artist", `${track?.liked ? "♥ " : ""}${pick.artist}`),
        el("span", "card-reason", pick.reason),
      );
      card.setAttribute("aria-label", `${i + 1}. ${pick.title} by ${pick.artist}, confidence ${confidence.toFixed(2)}. ${pick.reason}`);
      linkToStar(card, pick.deezer_id, onHighlight);
      card.addEventListener("click", () => track && onOpen(track));
      const item = el("li");
      item.append(card);
      return item;
    }));
    onAnswers(body.recommendations.map((pick) => pick.deezer_id));
  }

  return { reset };
}
