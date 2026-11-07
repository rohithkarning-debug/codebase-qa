/* Codebase QA - single-page front end (no build step, no framework). */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const state = {
  config: null,
  repos: [],
  repo: localStorage.getItem("cqa.repo") || "",
  mode: "agent",
  chunker: "ast",
  retriever: "code",
  questions: [],
  fileCache: new Map(),
  evalCache: null,
};

const LABELS = {
  ast: "Function chunks", fixed: "Fixed windows",
  code: "Code model", general: "General model", bm25: "Keyword (BM25)", hybrid: "Hybrid",
  agent: "Agent", rag: "Single retrieval",
};
const MODE_HINT = {
  agent: "The agent decides its own steps - semantic search, grep, opening files - and then answers with links to the exact lines.",
  rag: "Retrieves the six most similar chunks once and answers from them, without exploring further.",
};

/* ------------------------------------------------------------- utilities */
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* not json */ }
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return r.json();
}
function toast(msg, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}
const pct = (x) => (x == null ? "-" : `${Math.round(100 * x)}%`);
const basename = (p) => p.split("/").pop();
function shuffle(a) { const b = [...a]; for (let i = b.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [b[i], b[j]] = [b[j], b[i]]; } return b; }

/* ----------------------------------------------------- citation chips */
const EXT = "py|js|jsx|mjs|ts|tsx|java|go|rs|rb|c|h|cpp|cc|hpp|cs|php|kt";
const CITE_RE = new RegExp(
  `\\[?\`?([A-Za-z0-9_][\\w./-]*\\.(?:${EXT}))\\b\`?(?:\\s*(?::|#)\\s*L?(\\d+)(?:\\s*(?:-|–|to)\\s*L?(\\d+))?)?\`?\\]?`, "g");

function citeChip(repo, path, start, end, valid = true) {
  const lines = start ? (end && end !== start ? `L${start}–${end}` : `L${start}`) : "";
  return `<button type="button" class="cite${valid ? "" : " invalid"}" data-repo="${esc(repo)}" data-path="${esc(path)}"`
    + ` data-start="${start || ""}" data-end="${end || ""}" title="${esc(path)}${lines ? " " + lines : ""}">`
    + `<span>${esc(basename(path))}</span>${lines ? `<span class="ln">${lines}</span>` : ""}</button>`;
}

/** Replace path:Lx-Ly mentions in already-escaped text with chips. */
function linkCitations(html, repo, known) {
  return html.replace(CITE_RE, (m, path, s, e) => {
    const resolved = resolvePath(path, known);
    if (!s && !resolved) return m;           // bare file name we don't know: leave as text
    return citeChip(repo, resolved || path, s ? +s : null, e ? +e : (s ? +s : null), !!resolved || !known);
  });
}
function resolvePath(path, known) {
  if (!known || !known.length) return null;
  if (known.includes(path)) return path;
  const cands = known.filter((k) => k.endsWith("/" + path));
  return cands.length === 1 ? cands[0] : null;
}

/* ------------------------------------------------------------ markdown */
function renderMarkdown(text, repo, citations = []) {
  const known = [...new Set(citations.filter((c) => c.valid).map((c) => c.path))];
  const holds = [];
  const hold = (html) => `\u0000${holds.push(html) - 1}\u0000`;
  let src = String(text || "").replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
    hold(`<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`));
  src = esc(src);
  src = src.replace(/`([^`\n]+)`/g, (m, code) => {
    const whole = code.match(new RegExp(`^${CITE_RE.source}$`));
    if (whole && (whole[2] || resolvePath(whole[1], known))) return hold(linkCitations(code, repo, known));
    return hold(`<code>${code}</code>`);
  });
  src = linkCitations(src, repo, known);
  src = src.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
           .replace(/(^|[\s(])_([^_\n]+)_(?=[\s.,;:)]|$)/g, "$1<em>$2</em>");

  const out = [];
  let list = null;
  let para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${para.join(" ")}</p>`); para = []; } };
  const flushList = () => { if (list) { out.push(`<${list.tag}>${list.items.map((i) => `<li>${i}</li>`).join("")}</${list.tag}>`); list = null; } };
  for (const raw of src.split("\n")) {
    const line = raw.trimEnd();
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const h = line.match(/^#{1,4}\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const tag = ul ? "ul" : "ol";
      if (!list || list.tag !== tag) { flushList(); list = { tag, items: [] }; }
      list.items.push((ul || ol)[1]);
    } else if (h) {
      flushPara(); flushList(); out.push(`<p><strong>${h[1]}</strong></p>`);
    } else if (!line.trim()) {
      flushPara(); flushList();
    } else if (list && /^\s{2,}\S/.test(raw)) {
      list.items[list.items.length - 1] += " " + line.trim();
    } else {
      flushList(); para.push(line);
    }
  }
  flushPara(); flushList();
  return out.join("").replace(/\u0000(\d+)\u0000/g, (_, i) => holds[+i]);
}

/* --------------------------------------------------- syntax highlighting */
const KW = {
  python: "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield self cls match case",
  c: "abstract async await break case catch class const continue default delete do else enum export extends false final finally for fn func function go if impl implements import in instanceof interface let mod mut new nil null package private protected pub public return static struct super switch this throw throws trait true try type typeof use var void while yield def end module require",
};
function tokenRegex(lang) {
  if (lang === "python") {
    return /(?<com>#[^\n]*)|(?<str>[rbfuRBFU]{0,2}(?:"""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))|(?<deco>@[\w.]+)|(?<num>\b\d[\d_]*(?:\.\d+)?\b)|(?<id>[A-Za-z_]\w*)/g;
  }
  return /(?<com>\/\/[^\n]*|\/\*[\s\S]*?\*\/|#[^\n]*)|(?<str>`(?:\\.|[^`\\])*`|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')|(?<deco>@[\w.]+)|(?<num>\b\d[\d_]*(?:\.\d+)?\b)|(?<id>[A-Za-z_$][\w$]*)/g;
}
function highlight(src, lang) {
  const kws = new Set((KW[lang === "python" ? "python" : "c"]).split(" "));
  const re = tokenRegex(lang);
  let out = "", last = 0, prevWord = "", m;
  const wrap = (cls, text) => text.split("\n").map((part) => (part ? `<span class="${cls}">${esc(part)}</span>` : "")).join("\n");
  while ((m = re.exec(src))) {
    out += esc(src.slice(last, m.index));
    const g = m.groups;
    if (g.com) out += wrap("tok-com", g.com);
    else if (g.str) out += wrap("tok-str", g.str);
    else if (g.deco) out += wrap("tok-deco", g.deco);
    else if (g.num) out += wrap("tok-num", g.num);
    else if (g.id) {
      if (["def", "class", "function", "func", "fn", "struct", "interface"].includes(prevWord)) out += wrap("tok-def", g.id);
      else if (kws.has(g.id)) out += wrap("tok-kw", g.id);
      else out += esc(g.id);
      prevWord = g.id;
    }
    if (!g.id) prevWord = "";
    last = re.lastIndex;
  }
  out += esc(src.slice(last));
  return out.split("\n");
}

/* --------------------------------------------------------- code viewer */
async function openViewer(repo, path, start, end) {
  const key = `${repo}::${path}`;
  const viewer = $("#viewer");
  viewer.hidden = false;
  document.body.classList.add("viewer-open");
  $("#viewerPath").textContent = path;
  $("#viewerRange").textContent = start ? (end && end !== start ? `lines ${start}–${end}` : `line ${start}`) : "";
  try {
    let file = state.fileCache.get(key);
    if (!file) {
      $("#viewerBody").innerHTML = `<div class="viewer-empty"><span class="spinner"></span></div>`;
      file = await api(`/api/repos/${encodeURIComponent(repo)}/file?path=${encodeURIComponent(path)}`);
      file.html = highlight(file.content, file.language);
      state.fileCache.set(key, file);
    }
    const gh = file.url ? file.url + (start ? `#L${start}${end && end !== start ? `-L${end}` : ""}` : "") : "";
    const link = $("#viewerGithub");
    link.hidden = !gh;
    link.href = gh || "#";
    const s = start || 0, e = end || start || 0;
    const rows = file.html.map((line, i) => {
      const n = i + 1;
      const hl = n >= s && n <= e ? ` class="hl sweep" style="animation-delay:${Math.min(n - s, 20) * 12}ms"` : "";
      return `<tr id="L${n}"${hl}><td class="n">${n}</td><td class="c">${line || " "}</td></tr>`;
    }).join("");
    const body = $("#viewerBody");
    body.innerHTML = `<table class="code"><tbody>${rows}</tbody></table>`;
    const target = body.querySelector(`#L${Math.max(1, s - 4)}`);
    if (target) body.scrollTop = target.offsetTop - 8; else body.scrollTop = 0;
  } catch (err) {
    $("#viewerBody").innerHTML = `<div class="viewer-empty">Could not open ${esc(path)}: ${esc(err.message)}</div>`;
  }
}
function closeViewer() {
  $("#viewer").hidden = true;
  document.body.classList.remove("viewer-open");
}

document.addEventListener("click", (ev) => {
  const chip = ev.target.closest(".cite");
  if (chip && !chip.classList.contains("invalid")) {
    ev.preventDefault();
    openViewer(chip.dataset.repo, chip.dataset.path, +chip.dataset.start || null, +chip.dataset.end || null);
  }
});
document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") closeViewer(); });
$("#viewerClose").addEventListener("click", closeViewer);

/* ------------------------------------------------------------- top bar */
function renderModelStatus() {
  const llm = state.config.llm;
  const el = $("#modelStatus");
  el.classList.toggle("offline", llm.offline);
  if (llm.offline) {
    el.innerHTML = `<span class="dot"></span>Offline mode`;
    el.title = "No LLM API key found, so answers are assembled from search results. Add GEMINI_API_KEY or GROQ_API_KEY to .env and restart.";
  } else {
    el.innerHTML = `<span class="dot"></span><b>${esc(llm.model)}</b>`;
    el.title = `Answers by ${llm.name}`;
  }
}
function renderRepoSelect() {
  const sel = $("#repoSelect");
  sel.innerHTML = state.repos.map((r) => {
    const ready = r.cloned && r.indexes && Object.keys(r.indexes).length;
    return `<option value="${esc(r.name)}">${esc(r.name)}${ready ? "" : " (not indexed)"}</option>`;
  }).join("");
  if (!state.repos.some((r) => r.name === state.repo)) state.repo = state.repos[0]?.name || "";
  sel.value = state.repo;
}
$("#repoSelect").addEventListener("change", (e) => {
  state.repo = e.target.value;
  localStorage.setItem("cqa.repo", state.repo);
  renderExamples();
  fillCompareQuestions();
});

/* ---------------------------------------------------------------- tabs */
function showTab(name) {
  if (!["ask", "compare", "eval", "repos"].includes(name)) name = "ask";
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach((t) => (t.hidden = t.id !== `tab-${name}`));
  if (name === "eval") renderEval();
  if (name === "repos") renderRepos();
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

/* ----------------------------------------------------------------- ask */
function bindSeg(id, key, after) {
  $$(`#${id} button`).forEach((b) => b.addEventListener("click", () => {
    $$(`#${id} button`).forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    state[key] = b.dataset.value;
    if (after) after();
  }));
}
bindSeg("modeSeg", "mode", () => ($("#modeHint").textContent = MODE_HINT[state.mode]));
bindSeg("chunkSeg", "chunker");
$("#retrieverSel").addEventListener("change", (e) => (state.retriever = e.target.value));
$("#modeHint").textContent = MODE_HINT.agent;

function renderExamples() {
  const qs = state.questions.filter((q) => q.repo === state.repo);
  const box = $("#examples");
  if (!qs.length) {
    box.innerHTML = ["Where is authentication handled?", "Explain the main entry point.", "How are errors handled?"]
      .map((q) => `<button type="button" data-q="${esc(q)}">${esc(q)}</button>`).join("");
  } else {
    box.innerHTML = shuffle(qs).slice(0, 4).map((q) => `<button type="button" data-q="${esc(q.question)}">${esc(q.question)}</button>`).join("")
      + `<button type="button" class="shuffle" data-shuffle="1">More examples</button>`;
  }
}
$("#examples").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.shuffle) return renderExamples();
  $("#question").value = b.dataset.q;
  ask(b.dataset.q);
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#askForm").requestSubmit(); }
});
$("#askForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("#question").value.trim();
  if (q.length >= 3) ask(q);
});

function stepInput(action, input) {
  if (!input) return "";
  if (action === "search_code") return `“${esc(input.query || input._ || "")}”`;
  if (action === "grep") return `/${esc(input.pattern || input._ || "")}/${input.path ? " in " + esc(input.path) : ""}`;
  if (action === "find_symbol") return esc(input.name || input._ || "");
  if (action === "list_dir") return esc(input.path || input._ || "/");
  return esc(JSON.stringify(input));
}
function linkObservation(text, repo) {
  let html = esc(text);
  html = html.replace(/([\w./-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|rb|c|h|cpp|cs|php|kt)):L(\d+)-L(\d+)/g,
    (m, p, s, e) => citeChip(repo, p, +s, +e));
  html = html.replace(/^([\w./-]+\.(?:py|js|jsx|ts|tsx|java|go|rs|rb|c|h|cpp|cs|php|kt)):(\d+):/gm,
    (m, p, n) => citeChip(repo, p, +n, +n) + ":");
  return html;
}

async function ask(question) {
  if (!state.repo) return toast("Add a repository first (Repositories tab).");
  const btn = $("#askBtn");
  btn.disabled = true;
  const repo = state.repo;
  const art = document.createElement("article");
  art.className = "qa";
  art.innerHTML = `
    <h2 class="qa-q">${esc(question)}</h2>
    <div class="qa-meta">
      <span class="tag">${esc(repo)}</span><span class="tag">${LABELS[state.mode]}</span>
      <span class="tag">${LABELS[state.chunker]}</span><span class="tag">${LABELS[state.retriever]}</span>
    </div>
    <ol class="steps" hidden></ol>
    <div class="working"><span class="spinner"></span><span>${state.mode === "agent" ? "Searching the code" : "Retrieving code and writing the answer"}</span></div>
    <div class="answer-slot"></div>`;
  $("#thread").prepend(art);
  const steps = $(".steps", art), working = $(".working", art), slot = $(".answer-slot", art);
  const setWorking = (msg) => { working.hidden = !msg; if (msg) $("span:last-child", working).textContent = msg; };

  const handle = (ev) => {
    if (ev.type === "step") {
      steps.hidden = false;
      const li = document.createElement("li");
      li.className = "step";
      li.dataset.n = ev.step;
      const loc = ev.action === "open_file" && ev.locations?.[0]
        ? citeChip(repo, ev.locations[0].path, ev.locations[0].start, ev.locations[0].end) : stepInput(ev.action, ev.input);
      li.innerHTML = `
        <div class="step-head"><span class="tool">${esc(ev.action)}</span><span class="tool-input">${loc}</span></div>
        ${ev.thought ? `<p class="thought">${esc(ev.thought)}</p>` : ""}
        <details><summary>Result</summary><pre>${linkObservation(ev.observation || "", repo)}</pre></details>`;
      steps.append(li);
      setWorking("Deciding the next step");
    } else if (ev.type === "answer") {
      setWorking("");
      renderAnswer(slot, repo, ev);
      const first = (ev.citations || []).find((c) => c.valid && c.start);
      if (first) openViewer(repo, first.path, first.start, first.end);
    } else if (ev.type === "error") {
      setWorking("");
      slot.innerHTML = `<div class="answer error"><p><strong>Could not answer.</strong> ${esc(ev.message)}</p></div>`;
    }
  };

  try {
    const res = await fetch("/api/ask", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo, question, mode: state.mode, chunker: state.chunker, retriever: state.retriever }),
    });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, i);
        buf = buf.slice(i + 1);
        if (line.trim()) handle(JSON.parse(line));
      }
    }
    if (buf.trim()) handle(JSON.parse(buf));
  } catch (err) {
    handle({ type: "error", message: err.message });
  } finally {
    btn.disabled = false;
  }
}

function renderAnswer(slot, repo, ev) {
  const cites = ev.citations || [];
  const sources = cites.filter((c) => c.valid).map((c) =>
    citeChip(repo, c.path, c.start, c.end) + (c.url ? `<a class="icon-btn" href="${esc(c.url)}" target="_blank" rel="noopener" title="Open on GitHub" aria-label="Open ${esc(c.path)} on GitHub"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M6 3H3v10h10v-3M9 2h5v5M14 2 7 9" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg></a>` : "")).join("");
  const facts = [ev.steps ? `${ev.steps} tool call${ev.steps === 1 ? "" : "s"}` : null, ev.seconds != null ? `${ev.seconds}s` : null, ev.llm].filter(Boolean).join(", ");
  slot.innerHTML = `<div class="answer">${renderMarkdown(ev.answer, repo, cites)}
    <div class="answer-foot">${sources ? `<span>Sources</span>${sources}` : "<span>No file citations in this answer.</span>"}
    <span style="margin-left:auto">${esc(facts)}</span></div></div>`;
}

