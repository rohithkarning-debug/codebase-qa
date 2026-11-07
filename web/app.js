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

/* ------------------------------------------------------------- compare */
function fillCompareQuestions() {
  const sel = $("#compareQuestions");
  const qs = state.questions.filter((q) => q.repo === state.repo);
  sel.innerHTML = `<option value="">${qs.length ? "Test-set questions" : "No test-set questions for this repo"}</option>`
    + qs.map((q) => `<option value="${esc(q.question)}">${esc(q.id)}: ${esc(q.question)}</option>`).join("");
}
$("#compareQuestions").addEventListener("change", (e) => {
  if (!e.target.value) return;
  $("#compareQuery").value = e.target.value;
  $("#compareForm").requestSubmit();
});
$("#compareForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("#compareQuery").value.trim();
  if (q.length < 2) return;
  const grid = $("#compareGrid");
  grid.innerHTML = `<div class="working"><span class="spinner"></span><span>Searching 8 indexes</span></div>`;
  $("#compareGold").innerHTML = "";
  try {
    const data = await api(`/api/repos/${encodeURIComponent(state.repo)}/compare?q=${encodeURIComponent(q)}&k=5`);
    renderCompare(data);
  } catch (err) {
    grid.innerHTML = `<div class="answer error">${esc(err.message)}</div>`;
  }
});
function renderCompare(data) {
  const repo = state.repo;
  $("#compareGold").innerHTML = data.gold.length
    ? `Expected code: ${data.gold.map((g) => citeChip(repo, g.path, g.start, g.end) + ` <span>${esc(g.symbol)}</span>`).join(", ")}`
    : "Free-form question: no expected answer to score against.";
  const cells = [];
  for (const ch of state.config.chunkers) {
    for (const rt of state.config.retrievers) {
      const key = `${ch}+${rt}`;
      const hits = data.results[key];
      const color = ch === "ast" ? "var(--series-ast)" : "var(--series-fixed)";
      let verdict = "Top 5 results", ok = false;
      if (data.errors[key]) verdict = `Unavailable: ${data.errors[key]}`;
      else if (data.gold.length) {
        const r = hits.findIndex((h) => h.hit);
        ok = r >= 0;
        verdict = ok ? `First correct chunk at rank ${r + 1}` : "No correct chunk in the top 5";
      }
      const items = (hits || []).map((h) => `
        <li class="${h.hit ? "hit" : ""}" data-path="${esc(h.path)}" data-start="${h.start_line}" data-end="${h.end_line}" title="${esc(h.path)}:${h.start_line}-${h.end_line} (score ${h.score})">
          <span class="rk">${h.rank}</span>
          <span><span class="loc">${esc(basename(h.path))}:${h.start_line}–${h.end_line}</span><br><span class="sym">${esc(h.symbol || h.kind)}</span></span>
        </li>`).join("");
      cells.push(`<div class="cell"><h3><span class="swatch" style="background:${color}"></span>${LABELS[ch]} + ${LABELS[rt]}</h3>
        <p class="verdict${ok ? " ok" : ""}">${esc(verdict)}</p><ol>${items}</ol></div>`);
    }
  }
  $("#compareGrid").innerHTML = cells.join("");
}
$("#compareGrid").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-path]");
  if (li) openViewer(state.repo, li.dataset.path, +li.dataset.start, +li.dataset.end);
});

/* ---------------------------------------------------------------- eval */
async function renderEval(force = false) {
  const body = $("#evalBody");
  if (!state.evalCache || force) {
    body.innerHTML = `<div class="working"><span class="spinner"></span><span>Loading results</span></div>`;
    try { state.evalCache = await api("/api/eval/results"); } catch (err) { body.innerHTML = esc(err.message); return; }
  }
  const { retrieval, answers } = state.evalCache;
  if (!retrieval && !answers) {
    body.innerHTML = `<div class="empty"><h2 style="margin-top:0">No results yet</h2>
      <p>Run the evaluation from the project folder, then reload this page:</p>
      <pre>python -m cqa eval retrieval      # Recall@k for every chunking + embedding pair (no LLM needed)
python -m cqa eval answers        # RAG vs agent, LLM judge, file-location accuracy
python -m cqa eval report         # writes results/REPORT.md</pre></div>`;
    return;
  }
  let html = "";
  if (retrieval) html += retrievalHtml(retrieval);
  if (answers) html += answersHtml(answers);
  body.innerHTML = html;
  if (retrieval) drawRecallChart(retrieval, 5);
}

function heat(v, best) {
  return `<td class="num heat${best ? " best" : ""}" style="--a:${(0.06 + 0.32 * v).toFixed(2)}"><span>${pct(v)}</span></td>`;
}
function retrievalHtml(r) {
  const cfgs = r.configs;
  const names = Object.keys(cfgs).sort((a, b) => cfgs[b]["recall@5"] - cfgs[a]["recall@5"]);
  const best = names[0];
  const findings = [];
  if (best) findings.push(`Best index: <strong>${LABELS[best.split("+")[0]]} + ${LABELS[best.split("+")[1]]}</strong>, finding the right code in the top 5 for ${pct(cfgs[best]["recall@5"])} of questions.`);
  const deltas = (a, b) => state.config.retrievers.filter((x) => cfgs[`${a}+${x}`] && cfgs[`${b}+${x}`])
    .map((x) => cfgs[`${a}+${x}`]["recall@5"] - cfgs[`${b}+${x}`]["recall@5"]);
  const astD = deltas("ast", "fixed");
  if (astD.length) {
    const m = astD.reduce((s, x) => s + x, 0) / astD.length;
    findings.push(`Function/class chunks vs fixed windows: ${m >= 0 ? "+" : ""}${Math.round(100 * m)} points Recall@5 on average across retrievers.`);
  }
  const ce = ["ast", "fixed"].filter((c) => cfgs[`${c}+code`] && cfgs[`${c}+general`])
    .map((c) => `${LABELS[c].toLowerCase()} ${pct(cfgs[`${c}+code`]["recall@5"])} vs ${pct(cfgs[`${c}+general`]["recall@5"])}`);
  if (ce.length) findings.push(`Code model vs general model (Recall@5): ${ce.join("; ")}.`);
  const note = r.embed_backend === "hash"
    ? `<p class="note">These numbers come from the offline hashing embedder, not the real models. Run <code>python -m cqa eval retrieval</code> with internet access for the final results.</p>` : "";
  const ks = r.ks;
  const rows = names.map((n) => {
    const s = cfgs[n];
    const [ch, rt] = n.split("+");
    return `<tr><td><span class="swatch" style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;background:${ch === "ast" ? "var(--series-ast)" : "var(--series-fixed)"}"></span>${LABELS[ch]} + ${LABELS[rt]}</td>`
      + ks.map((k) => heat(s[`recall@${k}`], n === best)).join("")
      + `<td class="num">${s["mrr@10"].toFixed(3)}</td>${heat(s["file_acc@1"], false)}</tr>`;
  }).join("");
  const repos = Object.keys(cfgs[best]?.by_repo || {});
  const repoRows = names.map((n) => `<tr><td>${LABELS[n.split("+")[0]]} + ${LABELS[n.split("+")[1]]}</td>${repos.map((rp) => heat(cfgs[n].by_repo[rp]?.["recall@5"] ?? 0, false)).join("")}</tr>`).join("");
  const emb = r.embedders || {};
  return `
    <h2>Retrieval</h2>${note}
    <ul class="findings">${findings.map((f) => `<li>${f}</li>`).join("")}</ul>
    <div class="chart-card" style="margin:14px 0">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <h3 style="margin:0">Recall@<span id="chartK">5</span> by retriever</h3>
        <div class="seg" id="kSeg">${ks.map((k) => `<button type="button" data-k="${k}" aria-checked="${k === 5}">k = ${k}</button>`).join("")}</div>
        <div class="legend" style="margin-left:auto"><span><i style="background:var(--series-ast)"></i>Function chunks</span><span><i style="background:var(--series-fixed)"></i>Fixed windows</span></div>
      </div>
      <div id="recallChart"></div>
    </div>
    <div class="table-wrap"><table class="data">
      <thead><tr><th>Index</th>${ks.map((k) => `<th>Recall@${k}</th>`).join("")}<th>MRR@10</th><th>Top-1 file</th></tr></thead>
      <tbody>${rows}</tbody></table></div>
    <p class="hint">Code model: <code>${esc(emb.code?.model || "")}</code>. General model: <code>${esc(emb.general?.model || "")}</code>. A chunk counts as correct when it is in the expected file and covers at least half of the expected span (or the span covers half of it).</p>
    <h3>Recall@5 by repository</h3>
    <div class="table-wrap"><table class="data"><thead><tr><th>Index</th>${repos.map((x) => `<th>${esc(x)}</th>`).join("")}</tr></thead><tbody>${repoRows}</tbody></table></div>`;
}

function drawRecallChart(r, k) {
  const el = $("#recallChart");
  if (!el) return;
  $("#chartK").textContent = k;
  const rets = state.config.retrievers.filter((x) => r.configs[`ast+${x}`] || r.configs[`fixed+${x}`]);
  const W = 720, H = 240, L = 40, B = 28, T = 18, plotH = H - B - T, groupW = (W - L) / rets.length, barW = Math.min(46, groupW / 3.2);
  const y = (v) => T + plotH * (1 - v);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Recall at ${k} for each retriever, function chunks versus fixed windows">`;
  svg += `<g class="grid axis">${[0, 0.25, 0.5, 0.75, 1].map((v) => `<line x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`).join("")}</g>`;
  rets.forEach((rt, i) => {
    const cx = L + groupW * i + groupW / 2;
    [["ast", -1], ["fixed", 1]].forEach(([ch, side]) => {
      const s = r.configs[`${ch}+${rt}`];
      if (!s) return;
      const v = s[`recall@${k}`];
      const x = cx + (side < 0 ? -barW - 1 : 1);
      const h = Math.max(0, y(0) - y(v));
      const rr = Math.min(4, h);
      svg += `<path class="bar-${ch}" data-tip="${LABELS[ch]} + ${LABELS[rt]}: Recall@${k} ${pct(v)}" d="M${x},${y(0)} v${-(h - rr)} q0,${-rr} ${rr},${-rr} h${barW - 2 * rr} q${rr},0 ${rr},${rr} v${h - rr} z"/>`;
      svg += `<text class="val" x="${x + barW / 2}" y="${y(v) - 5}" text-anchor="middle">${pct(v)}</text>`;
    });
    svg += `<text class="lbl" x="${cx}" y="${H - 8}" text-anchor="middle">${LABELS[rt]}</text>`;
  });
  el.innerHTML = svg + "</svg>";
  $$("#kSeg button").forEach((b) => (b.onclick = () => {
    $$("#kSeg button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
    drawRecallChart(r, +b.dataset.k);
  }));
}
// chart tooltips
(() => {
  let tip = null;
  document.addEventListener("mousemove", (e) => {
    const t = e.target.closest?.("[data-tip]");
    if (!t) { if (tip) tip.hidden = true; return; }
    if (!tip) { tip = document.createElement("div"); tip.className = "chart-tip"; document.body.append(tip); }
    tip.hidden = false;
    tip.textContent = t.dataset.tip;
    tip.style.left = `${e.clientX + 12}px`;
    tip.style.top = `${e.clientY - 30}px`;
  });
})();

function answersHtml(a) {
  const modes = Object.keys(a.modes);
  const offline = /offline|heuristic/.test(a.llm + a.judge);
  const rows = modes.map((m) => {
    const s = a.modes[m];
    return `<tr><td>${LABELS[m] || m}</td>${heat(s.judge_correct)}<td class="num">${s.judge_score.toFixed(2)}</td>${heat(s.file_acc)}${heat(s.line_acc)}${heat(s.top_file_acc)}<td class="num">${s.avg_steps.toFixed(1)}</td><td class="num">${s.avg_seconds.toFixed(1)}s</td></tr>`;
  }).join("");
  const groupTable = (key, title) => {
    const groups = [...new Set(modes.flatMap((m) => Object.keys(a.modes[m][key])))].sort();
    return `<div><h3>${title}</h3><div class="table-wrap"><table class="data"><thead><tr><th>Mode</th>${groups.map((g) => `<th>${esc(g)}</th>`).join("")}</tr></thead><tbody>
      ${modes.map((m) => `<tr><td>${LABELS[m] || m}</td>${groups.map((g) => heat(a.modes[m][key][g]?.judge_correct ?? 0)).join("")}</tr>`).join("")}
      </tbody></table></div></div>`;
  };
  const byId = {};
  for (const r of a.rows) (byId[r.id] ||= { id: r.id, repo: r.repo, question: r.question, gold: r.gold, modes: {} }).modes[r.mode] = r;
  const qrows = Object.values(byId).map((q) => `
    <tr class="qrow" data-id="${esc(q.id)}"><td>${esc(q.id)}</td><td style="text-align:left;white-space:normal">${esc(q.question)}</td>
    ${modes.map((m) => { const r = q.modes[m]; if (!r) return "<td>-</td><td>-</td>";
      return `<td class="num">${r.judge.correct ? '<span class="mark-ok">✓</span>' : '<span class="mark-bad">✗</span>'} ${r.judge.score}</td><td class="num">${r.cite.file_hit ? '<span class="mark-ok">✓</span>' : '<span class="mark-bad">✗</span>'}</td>`; }).join("")}</tr>`).join("");
  state.evalRows = byId;
  return `
    <h2>Answers</h2>
    ${offline ? `<p class="note">Produced in offline mode, so the judge is a keyword heuristic. Add an LLM key to <code>.env</code> and run <code>python -m cqa eval answers</code> for the LLM-as-Judge scores.</p>` : ""}
    <p class="hint">Answer model <code>${esc(a.llm)}</code>, judge <code>${esc(a.judge)}</code>, index <code>${esc(a.index)}</code>, ${a.n_questions} questions.</p>
    <div class="table-wrap"><table class="data"><thead><tr><th>Mode</th><th>Judged correct</th><th>Judge score (1-5)</th><th>Cites right file</th><th>Cites right lines</th><th>First citation right</th><th>Tool calls</th><th>Time</th></tr></thead>
    <tbody>${rows}</tbody></table></div>
    <div class="two-col">${groupTable("by_repo", "Judged correct by repository")}${groupTable("by_type", "Judged correct by question type")}</div>
    <h3>Every question</h3>
    <p class="hint">Select a row to read the answers, the judge's reasoning and the expected code.</p>
    <div class="table-wrap"><table class="data" id="qtable"><thead><tr><th>ID</th><th style="text-align:left">Question</th>${modes.map((m) => `<th>${LABELS[m] || m}: judge</th><th>file</th>`).join("")}</tr></thead><tbody>${qrows}</tbody></table></div>`;
}
document.addEventListener("click", (e) => {
  const tr = e.target.closest?.(".qrow");
  if (!tr || e.target.closest(".cite")) return;
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("qdetail")) { next.remove(); return; }
  const q = state.evalRows[tr.dataset.id];
  const cols = tr.children.length;
  const parts = Object.entries(q.modes).map(([m, r]) => `
    <p><strong>${LABELS[m] || m}</strong>: judge ${r.judge.score}/5 - ${esc(r.judge.reason)}</p>
    <div class="answer">${renderMarkdown(r.answer, q.repo, r.citations)}</div>`).join("");
  const gold = q.gold.map((g) => citeChip(q.repo, g.path, g.start, g.end) + ` ${esc(g.symbol)}`).join(", ");
  const row = document.createElement("tr");
  row.className = "qdetail";
  row.innerHTML = `<td colspan="${cols}"><p>Expected code: ${gold}</p>${parts}</td>`;
  tr.after(row);
});

/* --------------------------------------------------------------- repos */
async function loadRepos() {
  state.repos = await api("/api/repos");
  renderRepoSelect();
}
function renderRepos() {
  const list = $("#repoList");
  list.innerHTML = state.repos.map((r) => {
    const idx = r.indexes || {};
    const pills = [];
    pills.push(`<span class="pill ${r.cloned ? "ok" : "missing"}">${r.cloned ? `Cloned at ${esc(r.commit || r.ref)}` : "Not cloned"}</span>`);
    for (const ch of state.config.chunkers) {
      const i = idx[ch];
      pills.push(`<span class="pill ${i ? "ok" : "missing"}">${LABELS[ch]}${i ? `: ${i.n_chunks} chunks` : ": not built"}</span>`);
    }
    for (const em of Object.keys(state.config.embedders)) {
      const has = Object.values(idx).some((i) => i.dense.includes(em));
      pills.push(`<span class="pill ${has ? "ok" : "missing"}">${LABELS[em]} embeddings</span>`);
    }
    return `<div class="repo">
      <div><span class="name">${esc(r.name)}</span> <a href="${esc(r.url)}" target="_blank" rel="noopener" class="desc">${esc(r.url.replace("https://github.com/", ""))}${r.ref ? "@" + esc(r.ref) : ""}</a>
      ${r.eval_repo ? '<span class="tag">test set</span>' : ""}<div class="desc">${esc(r.description || "")}</div></div>
      <button class="secondary" data-index="${esc(r.name)}">Build indexes</button>
      <div class="facts">${pills.join("")}</div></div>`;
  }).join("");
}
$("#repoList").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-index]");
  if (!b) return;
  b.disabled = true;
  try { trackJob(await api(`/api/repos/${encodeURIComponent(b.dataset.index)}/index`, { method: "POST" })); }
  catch (err) { toast(err.message); b.disabled = false; }
});
$("#addRepoForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const job = await api("/api/repos", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: $("#repoUrl").value.trim(), ref: $("#repoRef").value.trim() }),
    });
    $("#repoUrl").value = "";
    $("#repoRef").value = "";
    trackJob(job);
  } catch (err) { toast(err.message); }
});
function trackJob(job) {
  const box = document.createElement("div");
  box.className = "job";
  $("#jobs").prepend(box);
  const tick = async () => {
    try { job = await api(`/api/jobs/${job.id}`); } catch (_) { return; }
    const p = job.total ? Math.round((100 * job.done) / job.total) : (job.status === "done" ? 100 : 4);
    box.classList.toggle("error", job.status === "error");
    const label = job.status === "done" ? `Indexed ${job.repo}. You can ask questions about it now.`
      : job.status === "error" ? `Indexing ${job.repo} failed: ${job.message}`
      : `${job.repo}: ${job.stage || "starting"}${job.total ? ` (${job.done}/${job.total})` : ""}`;
    box.innerHTML = `<div>${esc(label)}</div>${job.status === "error" ? "" : `<div class="bar"><div style="width:${p}%"></div></div>`}`;
    if (job.status === "done" || job.status === "error") {
      await loadRepos();
      renderRepos();
      if (job.status === "done") { state.repo = job.repo; renderRepoSelect(); renderExamples(); fillCompareQuestions(); }
      if (job.status === "done") setTimeout(() => box.remove(), 6000);
      return;
    }
    setTimeout(tick, 1000);
  };
  tick();
}

/* ---------------------------------------------------------------- boot */
(async function boot() {
  try {
    [state.config, state.questions] = await Promise.all([api("/api/config"), api("/api/eval/questions")]);
    await loadRepos();
    renderModelStatus();
    renderExamples();
    fillCompareQuestions();
    showTab(location.hash.slice(1) || "ask");
  } catch (err) {
    document.querySelector("main").innerHTML = `<div class="answer error"><p><strong>The server did not respond.</strong> Start it with <code>python -m cqa serve</code> and reload. (${esc(err.message)})</p></div>`;
  }
})();
