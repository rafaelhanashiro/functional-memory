// The literature list (literature/papers.json) is shown in two groups, seen and not seen
// yet, newest first. Adding a link, switching a resource between the groups, and saving a
// write-up all go through serve.py. Each write-up is a Markdown file (math in $...$,
// $$...$$, \(...\), \[...\]) fetched whenever its item opens; "Edit" shows it beside a live
// preview. Code spans and blocks are matched first so their dollar signs stay literal.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;
const EMPTY = '<p class="empty">No write-up yet.</p>';
const API = "api/literature";
const CHECK = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 10.5l3 3 6-7"/></svg>';
const CHEVRON = '<svg class="chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 7.5l5 5 5-5"/></svg>';
const groups = {
  seen: document.querySelector('[data-group="seen"]'),
  unseen: document.querySelector('[data-group="unseen"]'),
};
const form = document.querySelector(".add-resource");
const status = document.querySelector(".literature-status");
const cards = new Map();
const dirty = new Set();
let papers = [];

function render(markdown) {
  const math = [];
  const source = markdown.replace(MATH, (match, code, display, bracket, paren, inline) => {
    if (code !== undefined) return code;
    const block = display !== undefined || bracket !== undefined;
    math.push(katex.renderToString(display ?? bracket ?? paren ?? inline, {displayMode: block, throwOnError: false}));
    return `${math.length - 1}`;
  });
  return marked.parse(source).replace(/(\d+)/g, (match, index) => math[index]);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(label, kind, onClick) {
  const node = element("button", `button ${kind}`, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

async function failure(response) {
  try {
    return (await response.json()).error;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

function authorsLabel(authors) {
  const surname = (name) => name.trim().split(/\s+/).pop();
  if (authors.length === 0) return "";
  if (authors.length === 1) return surname(authors[0]);
  if (authors.length === 2) return `${surname(authors[0])} and ${surname(authors[1])}`;
  return `${surname(authors[0])} et al.`;
}

function writeupPath(paper) {
  return `literature/${paper.id}.md`;
}

// Write-ups: a rendered view with "Edit", or the editor with a live preview.

function showView(card, paper, text) {
  dirty.delete(card);
  const view = element("div", "writeup-view");
  view.innerHTML = text.trim() ? render(text) : EMPTY;
  card.querySelector(".writeup").replaceChildren(button("Edit", "edit", () => showEditor(card, paper, text)), view);
}

function showEditor(card, paper, original) {
  const input = element("textarea", "editor-input");
  input.value = original;
  input.placeholder = "Markdown, with math in $...$ and $$...$$";
  const preview = element("div", "writeup-view editor-preview");
  const note = element("span", "editor-status");
  const saveButton = button("Save", "primary", () => save());
  let frame = 0;

  const update = () => { preview.innerHTML = input.value.trim() ? render(input.value) : EMPTY; };
  const grow = () => {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight + 2}px`;
  };
  const cancel = () => {
    if (!dirty.has(card) || confirm("Discard unsaved changes?")) showView(card, paper, original);
  };
  async function save() {
    const text = input.value.trim() ? `${input.value.trimEnd()}\n` : "";
    saveButton.disabled = true;
    note.textContent = "Saving…";
    try {
      const response = await fetch(writeupPath(paper), {
        method: "PUT", body: text, headers: {"Content-Type": "text/markdown; charset=utf-8"},
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      showView(card, paper, text);
    } catch (error) {
      note.textContent = `Not saved: ${error.message}`;
      saveButton.disabled = false;
    }
  }

  input.addEventListener("input", () => {
    if (input.value === original) dirty.delete(card);
    else dirty.add(card);
    note.textContent = dirty.has(card) ? "Unsaved changes" : "";
    grow();
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(update);
  });
  input.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      save();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });

  const panes = element("div", "editor-panes");
  panes.append(input, preview);
  const actions = element("div", "editor-actions");
  actions.append(note, button("Cancel", "ghost", cancel), saveButton);
  const editor = element("div", "editor");
  editor.append(panes, actions);
  card.querySelector(".writeup").replaceChildren(editor);
  update();
  grow();
  input.focus();
}

// The list: one card per resource, placed in its group.

function createCard(paper) {
  const card = element("details", "paper");
  const summary = element("summary");
  const head = element("span", "paper-head");
  const meta = element("span", "paper-meta");
  for (const part of [authorsLabel(paper.authors ?? []), paper.year ? String(paper.year) : ""]) {
    if (part) meta.append(part, " · ");
  }
  const link = element("a", "", paper.source || "Link");
  link.href = paper.url;
  link.target = "_blank";
  link.rel = "noopener";
  meta.append(link);
  head.append(element("span", "paper-title", paper.title), meta);
  const toggle = element("button", "seen-toggle");
  toggle.type = "button";
  toggle.innerHTML = CHECK;
  toggle.addEventListener("click", (event) => {
    event.preventDefault();
    setSeen(paper, !paper.seen);
  });
  summary.append(head, toggle);
  summary.insertAdjacentHTML("beforeend", CHEVRON);
  summary.addEventListener("click", (event) => {
    if (event.target.closest("a, button") || !card.open || !dirty.has(card)) return;
    if (confirm("Discard unsaved changes?")) dirty.delete(card);
    else event.preventDefault();
  });
  card.append(summary, element("div", "writeup"));
  card.addEventListener("toggle", async () => {
    if (!card.open) return;
    try {
      const response = await fetch(writeupPath(paper), {cache: "no-store"});
      if (!response.ok) throw new Error(response.statusText);
      showView(card, paper, await response.text());
    } catch {
      card.querySelector(".writeup").innerHTML = '<p class="empty">Could not load the write-up.</p>';
    }
  });
  cards.set(paper.id, card);
  return card;
}

function layout() {
  const order = {seen: [], unseen: []};
  for (const paper of [...papers].reverse()) {
    const card = cards.get(paper.id) ?? createCard(paper);
    card.classList.toggle("seen", paper.seen);
    card.classList.toggle("unseen", !paper.seen);
    const toggle = card.querySelector(".seen-toggle");
    const label = paper.seen ? "Mark as not seen" : "Mark as seen";
    toggle.setAttribute("aria-pressed", String(paper.seen));
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
    order[paper.seen ? "seen" : "unseen"].push(card);
  }
  for (const [name, group] of Object.entries(groups)) {
    const list = group.querySelector(".papers");
    order[name].forEach((card, index) => {
      if (list.children[index] !== card) list.insertBefore(card, list.children[index] ?? null);
    });
    group.hidden = order[name].length === 0;
  }
}

function settle(card) {
  card.classList.remove("moved");
  void card.offsetWidth;
  card.classList.add("moved");
}

async function setSeen(paper, seen) {
  paper.seen = seen;
  layout();
  settle(cards.get(paper.id));
  try {
    const response = await fetch(`${API}/${encodeURIComponent(paper.id)}`, {
      method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({seen}),
    });
    if (!response.ok) throw new Error(await failure(response));
    status.textContent = "";
  } catch (error) {
    paper.seen = !seen;
    layout();
    status.textContent = `Not saved: ${error.message}`;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  let url = form.elements.url.value.trim();
  if (!/^https?:\/\//i.test(url)) url = `https://${url}`;
  for (const control of form.elements) control.disabled = true;
  status.textContent = "Adding…";
  try {
    const response = await fetch(API, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({url}),
    });
    if (!response.ok) throw new Error(await failure(response));
    const paper = await response.json();
    papers.push(paper);
    layout();
    settle(cards.get(paper.id));
    form.reset();
    status.textContent = "";
  } catch (error) {
    status.textContent = error.message;
  } finally {
    for (const control of form.elements) control.disabled = false;
    form.elements.url.focus();
  }
});

window.addEventListener("beforeunload", (event) => {
  if (dirty.size) event.preventDefault();
});

(async () => {
  try {
    const response = await fetch("literature/papers.json", {cache: "no-store"});
    if (!response.ok) throw new Error(response.statusText);
    papers = await response.json();
    layout();
  } catch {
    status.textContent = "Could not load the literature list.";
  }
})();
