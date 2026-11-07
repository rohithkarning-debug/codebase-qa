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

