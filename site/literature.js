// The literature list (literature/papers.json) is shown in its stored order, split into
// seen and not seen yet; new links go to the top and the grip on each card reorders them
// (drag, or arrow keys). Adding, reordering, switching groups, and saving a write-up all
// go through serve.py. Each write-up is a Markdown file (math in $...$,
// $$...$$, \(...\), \[...\]) fetched whenever its item opens; "Edit" shows it beside a live
// preview. Code spans and blocks are matched first so their dollar signs stay literal.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;
const EMPTY = '<p class="empty">No write-up yet.</p>';
const API = "api/literature";
const CHECK = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 10.5l3 3 6-7"/></svg>';
const CHEVRON = '<svg class="chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 7.5l5 5 5-5"/></svg>';
const GRIP = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="7.5" cy="5" r="1.4"/><circle cx="12.5" cy="5" r="1.4"/><circle cx="7.5" cy="10" r="1.4"/><circle cx="12.5" cy="10" r="1.4"/><circle cx="7.5" cy="15" r="1.4"/><circle cx="12.5" cy="15" r="1.4"/></svg>';
const groups = {
  seen: document.querySelector('[data-group="seen"]'),
  unseen: document.querySelector('[data-group="unseen"]'),
};
const form = document.querySelector(".add-resource");
const status = document.querySelector(".literature-status");
const cards = new Map();
const dirty = new Set();
let papers = [];
let saving = Promise.resolve();

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
  const title = element("input", "editor-title");
  title.value = paper.title;
  title.placeholder = "Title";
  title.setAttribute("aria-label", "Title");
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
  const mark = () => {
    if (title.value.trim() === paper.title && input.value === original) dirty.delete(card);
    else dirty.add(card);
    note.textContent = dirty.has(card) ? "Unsaved changes" : "";
  };
  const cancel = () => {
    if (!dirty.has(card) || confirm("Discard unsaved changes?")) showView(card, paper, original);
  };
  async function save() {
    const name = title.value.replace(/\s+/g, " ").trim();
    if (!name) {
      note.textContent = "The title can't be empty";
      title.focus();
      return;
    }
    const text = input.value === original ? original : input.value.trim() ? `${input.value.trimEnd()}\n` : "";
    saveButton.disabled = true;
    note.textContent = "Saving…";
    try {
      if (name !== paper.title) {
        const response = await fetch(`${API}/${encodeURIComponent(paper.id)}`, {
          method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({title: name}),
        });
        if (!response.ok) throw new Error(await failure(response));
        paper.title = name;
        card.querySelector(".paper-title").textContent = name;
      }
      if (text !== original) {
        const response = await fetch(writeupPath(paper), {
          method: "PUT", body: text, headers: {"Content-Type": "text/markdown; charset=utf-8"},
        });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      }
      showView(card, paper, text);
    } catch (error) {
      note.textContent = `Not saved: ${error.message}`;
      saveButton.disabled = false;
    }
  }

  title.addEventListener("input", mark);
  input.addEventListener("input", () => {
    mark();
    grow();
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(update);
  });

  const panes = element("div", "editor-panes");
  panes.append(input, preview);
  const actions = element("div", "editor-actions");
  actions.append(note, button("Cancel", "ghost", cancel), saveButton);
  const editor = element("div", "editor");
  editor.append(title, panes, actions);
  editor.addEventListener("keydown", (event) => {
    if (((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s")
        || (event.target === title && event.key === "Enter")) {
      event.preventDefault();
      save();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });
  card.querySelector(".writeup").replaceChildren(editor);
  update();
  grow();
  input.focus();
}

// The list: one card per resource, placed in its group.

function createCard(paper) {
  const card = element("details", "paper");
  card.dataset.id = paper.id;
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
  const handle = element("button", "drag-handle");
  handle.type = "button";
  handle.innerHTML = GRIP;
  handle.title = "Drag to reorder";
  handle.setAttribute("aria-label", "Reorder: drag, or press the up and down arrow keys");
  handle.addEventListener("click", (event) => event.preventDefault());
  handle.addEventListener("pointerdown", (event) => drag(event, card, handle));
  handle.addEventListener("keydown", (event) => nudge(event, card, handle));
  summary.append(handle, head, toggle);
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
  for (const paper of papers) {
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

// Reordering: the grip lifts its card, a dashed slot follows the pointer within the
// card's group, and the drop saves the new order of the whole list.

function drag(event, card, handle) {
  if (event.button !== 0) return;
  event.preventDefault();
  const list = card.parentElement;
  const start = card.getBoundingClientRect();
  const offset = event.clientY - start.top;
  const slot = element("div", "drop-slot");
  slot.style.height = `${start.height}px`;
  list.insertBefore(slot, card);
  Object.assign(card.style, {
    position: "fixed", left: `${start.left}px`, top: `${start.top}px`, width: `${start.width}px`, zIndex: "20",
  });
  card.classList.add("dragging");
  document.body.classList.add("reordering");
  handle.setPointerCapture(event.pointerId);
  let pointer = event.clientY;
  let frame = 0;

  const place = () => {
    card.style.top = `${pointer - offset}px`;
    let before = null;
    for (const sibling of list.children) {
      if (sibling === card || sibling === slot) continue;
      const box = sibling.getBoundingClientRect();
      if (pointer < box.top + box.height / 2) {
        before = sibling;
        break;
      }
    }
    list.insertBefore(slot, before);
  };
  const scroll = () => {
    const edge = 72;
    const speed = pointer < edge ? (pointer - edge) / 4 : pointer > innerHeight - edge ? (pointer - innerHeight + edge) / 4 : 0;
    if (speed) {
      window.scrollBy(0, speed);
      place();
    }
    frame = requestAnimationFrame(scroll);
  };
  const move = (moved) => {
    pointer = moved.clientY;
    place();
  };
  const end = () => {
    cancelAnimationFrame(frame);
    handle.removeEventListener("pointermove", move);
    handle.removeEventListener("pointerup", end);
    handle.removeEventListener("pointercancel", end);
    list.insertBefore(card, slot);
    slot.remove();
    card.removeAttribute("style");
    card.classList.remove("dragging");
    document.body.classList.remove("reordering");
    saveOrder(card);
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  frame = requestAnimationFrame(scroll);
}

function nudge(event, card, handle) {
  const up = event.key === "ArrowUp";
  if (!up && event.key !== "ArrowDown") return;
  event.preventDefault();
  const neighbor = up ? card.previousElementSibling : card.nextElementSibling;
  if (!neighbor) return;
  card.parentElement.insertBefore(card, up ? neighbor : neighbor.nextElementSibling);
  handle.focus();
  saveOrder(card);
}

function saveOrder(card) {
  const id = card.dataset.id;
  const next = card.nextElementSibling?.dataset.id;
  const previous = card.previousElementSibling?.dataset.id;
  const before = papers.map((paper) => paper.id);
  const order = before.filter((other) => other !== id);
  if (next) order.splice(order.indexOf(next), 0, id);
  else if (previous) order.splice(order.indexOf(previous) + 1, 0, id);
  else return;
  if (order.join(" ") === before.join(" ")) return;
  const byId = new Map(papers.map((paper) => [paper.id, paper]));
  papers = order.map((other) => byId.get(other));
  layout();
  saving = saving.then(async () => {
    try {
      const response = await fetch(API, {
        method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({order}),
      });
      if (!response.ok) throw new Error(await failure(response));
      status.textContent = "";
    } catch (error) {
      status.textContent = `Order not saved: ${error.message}`;
    }
  });
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
    papers.unshift(paper);
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
