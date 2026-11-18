// The literature list (literature/papers.json) is shown in its stored order, split into
// seen and not seen yet; new links go to the top and the grip on each card reorders them
// (drag, or arrow keys). Adding, reordering, switching groups, renaming, and saving a
// write-up (literature/<id>.md, shown when its card opens) all go through serve.py.
import {button, dirty, element, failure, saveText, showEditor, showView} from "./editor.js";

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
let papers = [];
let saving = Promise.resolve();

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

// Write-ups: a rendered view with "Edit", or the shared editor with the title field.

function openWriteup(card, paper, text) {
  showView(card.querySelector(".writeup"), text, {empty: EMPTY, onEdit: () => editWriteup(card, paper, text)});
}

function editWriteup(card, paper, text) {
  showEditor(card.querySelector(".writeup"), {
    key: card, text, title: paper.title, empty: EMPTY,
    save: async ({text: next, title}) => {
      if (title !== paper.title) {
        const response = await fetch(`${API}/${encodeURIComponent(paper.id)}`, {
          method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({title}),
        });
        if (!response.ok) throw new Error(await failure(response));
        paper.title = title;
        card.querySelector(".paper-title").textContent = title;
      }
      if (next !== text) await saveText(writeupPath(paper), next);
    },
    close: (next) => openWriteup(card, paper, next),
  });
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
      openWriteup(card, paper, await response.text());
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
