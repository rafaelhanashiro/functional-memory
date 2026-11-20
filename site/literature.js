// The literature list (literature/papers.json), shown in its stored order under the user's own
// categories (literature/categories.json, in their order), each split into seen and not seen yet;
// resources under no category come last, under no heading while there are no categories. A new
// link goes to the top of the category picked beside it. The grip on each card drags it within its
// list, into the other half, or into another category (the arrow keys move it within its list). A
// category's name renames in place, its arrows move it, and deleting it leaves its resources under
// no category. The chevron before a name hides that category down to its heading, in this browser
// only. Adding, moving, renaming, and saving a write-up (literature/<id>.md, shown when its
// card opens) all go through serve.py.
import {dirty, element, failure, saveText, showEditor, showView} from "./editor.js";

const EMPTY = '<p class="empty">No write-up yet.</p>';
const API = "api/literature";
const CATEGORIES = "api/categories";
const CHECK = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 10.5l3 3 6-7"/></svg>';
const CHEVRON = '<svg class="chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 7.5l5 5 5-5"/></svg>';
const GRIP = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="7.5" cy="5" r="1.4"/><circle cx="12.5" cy="5" r="1.4"/><circle cx="7.5" cy="10" r="1.4"/><circle cx="12.5" cy="10" r="1.4"/><circle cx="7.5" cy="15" r="1.4"/><circle cx="12.5" cy="15" r="1.4"/></svg>';
const UP = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 15V5M5.5 9.5L10 5l4.5 4.5"/></svg>';
const DOWN = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 5v10M5.5 10.5L10 15l4.5-4.5"/></svg>';
const CROSS = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 5.5l9 9M14.5 5.5l-9 9"/></svg>';
const HALVES = [["seen", "Seen"], ["unseen", "Not seen yet"]];
const PICKED = "literature-category";  // this browser's last pick for new links
const FOLDED = "literature-hidden";  // the categories this browser hides
const container = document.querySelector(".categories");
const form = document.querySelector(".add-resource");
const picker = form.elements.category;
const status = document.querySelector(".literature-status");
const cards = new Map();
const sections = new Map();  // by category id, "" for the resources under none
const folded = new Set((() => {
  try { return JSON.parse(localStorage.getItem(FOLDED) ?? "[]"); } catch { return []; }
})());
let papers = [];
let categories = [];
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

// A resource's category, or "" when it has none or names one that no longer exists.
const categoryOf = (paper) => categories.some((category) => category.id === paper.category) ? paper.category : "";

async function request(method, url, body) {
  const response = await fetch(url, {
    method, headers: {"Content-Type": "application/json"}, body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await failure(response));
  return response.status === 204 ? null : response.json();
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
        await request("PATCH", `${API}/${encodeURIComponent(paper.id)}`, {title});
        paper.title = title;
        card.querySelector(".paper-title").textContent = title;
      }
      if (next !== text) await saveText(writeupPath(paper), next);
    },
    close: (next) => openWriteup(card, paper, next),
  });
}

// The cards: one per resource.

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
  handle.title = "Drag to reorder, or into another category";
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

// The sections: one per category, in order, with its two halves, then the one for the resources
// under no category.

function icon(label, svg, onClick, kind = "") {
  const node = element("button", kind ? `category-icon ${kind}` : "category-icon");
  node.type = "button";
  node.title = label;
  node.setAttribute("aria-label", label);
  node.innerHTML = svg;
  node.addEventListener("click", onClick);
  return node;
}

function createSection(id) {
  const section = element("section", id ? "category" : "category uncategorized");
  section.dataset.category = id;
  const head = element("div", "category-head");
  const name = element("h2", "category-name", id ? "" : "No category");
  head.append(icon("", CHEVRON, () => fold(id, !folded.has(id)), "fold"), name, element("span", "category-count"));
  if (id) {
    name.tabIndex = 0;
    name.title = "Rename";
    name.addEventListener("click", () => renameCategory(id));
    name.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.target !== name) return;
      event.preventDefault();
      renameCategory(id);
    });
    const actions = element("span", "category-actions");
    actions.append(
      icon("Move up", UP, () => moveCategory(id, -1), "up"),
      icon("Move down", DOWN, () => moveCategory(id, 1), "down"),
      icon("Delete this category", CROSS, () => deleteCategory(id), "delete"),
    );
    head.append(actions);
  }
  section.append(head);
  for (const [half, label] of HALVES) {
    const group = element("div", "resource-group");
    group.dataset.group = half;
    group.append(element("h3", "", label), element("div", "papers"));
    section.append(group);
  }
  return section;
}

// A hidden category folds to its heading, which keeps its count; the choice stays in this browser.
function showFold(id) {
  const section = sections.get(id);
  const hidden = folded.has(id);
  const button = section.querySelector(".category-icon.fold");
  const label = hidden ? "Show this category" : "Hide this category";
  section.classList.toggle("folded", hidden);
  button.title = label;
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", String(!hidden));
}

function fold(id, hidden) {
  if (hidden) folded.add(id);
  else folded.delete(id);
  try { localStorage.setItem(FOLDED, JSON.stringify([...folded])); } catch { /* hiding is only a convenience */ }
  showFold(id);
}

function render() {
  const ids = [...categories.map((category) => category.id), ""];
  for (const id of [...sections.keys()]) if (!ids.includes(id)) sections.delete(id);
  container.replaceChildren(...ids.map((id, index) => {
    if (!sections.has(id)) sections.set(id, createSection(id));
    const section = sections.get(id);
    showFold(id);
    if (id) {
      const name = section.querySelector(".category-name");
      if (!name.querySelector("input")) name.textContent = categories[index].name;
      section.querySelector(".up").disabled = index === 0;
      section.querySelector(".down").disabled = index === categories.length - 1;
    }
    return section;
  }));
  document.body.classList.toggle("has-categories", categories.length > 0);
  fillPicker();
  layout();
}

// Each card goes to its category's half, in the stored order; a category shows both halves even
// when empty, and the resources under no category show only the halves they fill.
function layout() {
  const lists = new Map();
  for (const paper of papers) {
    const card = cards.get(paper.id) ?? createCard(paper);
    card.classList.toggle("seen", paper.seen);
    card.classList.toggle("unseen", !paper.seen);
    const toggle = card.querySelector(".seen-toggle");
    const label = paper.seen ? "Mark as not seen" : "Mark as seen";
    toggle.setAttribute("aria-pressed", String(paper.seen));
    toggle.setAttribute("aria-label", label);
    toggle.title = label;
    const key = `${categoryOf(paper)}/${paper.seen ? "seen" : "unseen"}`;
    if (!lists.has(key)) lists.set(key, []);
    lists.get(key).push(card);
  }
  for (const [id, section] of sections) {
    let total = 0;
    for (const group of section.querySelectorAll(".resource-group")) {
      const order = lists.get(`${id}/${group.dataset.group}`) ?? [];
      const list = group.querySelector(".papers");
      order.forEach((card, index) => {
        if (list.children[index] !== card) list.insertBefore(card, list.children[index] ?? null);
      });
      group.hidden = !id && order.length === 0;
      total += order.length;
    }
    section.querySelector(".category-count").textContent = total ? String(total) : "";
    if (!id) section.hidden = total === 0;
  }
}

// The picker beside the link: the categories, then none; it keeps this browser's last pick.
function fillPicker() {
  let chosen = picker.value;
  if (!picker.options.length) {
    try { chosen = localStorage.getItem(PICKED) ?? ""; } catch { chosen = ""; }
  }
  picker.replaceChildren(...categories.map((category) => new Option(category.name, category.id)), new Option("No category", ""));
  picker.value = categories.some((category) => category.id === chosen) ? chosen : "";
  picker.hidden = categories.length === 0;
}

picker.addEventListener("change", () => {
  try { localStorage.setItem(PICKED, picker.value); } catch { /* the pick is only a convenience */ }
});

// Categories: named in place (Enter or leaving the field saves, Escape cancels), moved by their
// arrows, and deleted after asking.

function editName(heading, value, save, cancel) {
  const input = element("input", "category-input");
  input.value = value;
  input.maxLength = 200;
  input.placeholder = "Category name";
  heading.replaceChildren(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = async (keep) => {
    if (finished) return;
    finished = true;
    const name = input.value.replace(/\s+/g, " ").trim();
    if (!keep || !name || name === value) {
      cancel();
      return;
    }
    input.disabled = true;
    try {
      await save(name);
      status.textContent = "";
    } catch (error) {
      status.textContent = error.message;
      cancel();
    }
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === "Escape") {
      event.preventDefault();
      finish(event.key === "Enter");
    }
  });
  input.addEventListener("blur", () => finish(true));
}

document.querySelector('[data-action="add-category"]').addEventListener("click", () => {
  const open = container.querySelector(".category.draft input");
  if (open) {
    open.focus();
    return;
  }
  const draft = element("section", "category draft");
  const head = element("div", "category-head");
  const heading = element("h2", "category-name");
  head.append(heading);
  draft.append(head);
  container.prepend(draft);
  editName(heading, "", async (name) => {
    const category = await request("POST", CATEGORIES, {name});
    draft.remove();
    categories.unshift(category);
    render();
  }, () => draft.remove());
});

function renameCategory(id) {
  const category = categories.find((other) => other.id === id);
  const heading = sections.get(id).querySelector(".category-name");
  if (!category || heading.querySelector("input")) return;
  editName(heading, category.name, async (name) => {
    const renamed = await request("PATCH", `${CATEGORIES}/${encodeURIComponent(id)}`, {name});
    category.name = renamed.name;
    heading.textContent = renamed.name;
    fillPicker();
  }, () => { heading.textContent = category.name; });
}

async function moveCategory(id, step) {
  const index = categories.findIndex((category) => category.id === id);
  const target = index + step;
  if (index < 0 || target < 0 || target >= categories.length) return;
  const previous = [...categories];
  categories.splice(target, 0, ...categories.splice(index, 1));
  render();
  sections.get(id).querySelector(step < 0 ? ".up" : ".down").focus();
  try {
    await request("PUT", CATEGORIES, {order: categories.map((category) => category.id)});
    status.textContent = "";
  } catch (error) {
    categories = previous;
    render();
    status.textContent = `Order not saved: ${error.message}`;
  }
}

async function deleteCategory(id) {
  const category = categories.find((other) => other.id === id);
  const count = papers.filter((paper) => categoryOf(paper) === id).length;
  const stay = count ? ` Its ${count} resource${count === 1 ? "" : "s"} stay${count === 1 ? "s" : ""}, under no category.` : "";
  if (!category || !confirm(`Delete the category "${category.name}"?${stay}`)) return;
  try {
    await request("DELETE", `${CATEGORIES}/${encodeURIComponent(id)}`);
    categories = categories.filter((other) => other.id !== id);
    for (const paper of papers) if (paper.category === id) delete paper.category;
    render();
    status.textContent = "";
  } catch (error) {
    status.textContent = error.message;
  }
}

// Reordering: the grip lifts its card and a dashed slot follows the pointer into the nearest
// list, its own or another half or category's; the drop files the card there and saves the new
// order of the whole list. While dragging, a category's empty halves (and, once there are
// categories, the resources under none) stay open to drop into.

function drag(event, card, handle) {
  if (event.button !== 0) return;
  event.preventDefault();
  const start = card.getBoundingClientRect();
  const offset = event.clientY - start.top;
  const slot = element("div", "drop-slot");
  slot.style.height = `${start.height}px`;
  let list = card.parentElement;
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
    let nearest = Infinity;
    for (const candidate of container.querySelectorAll(".papers")) {
      if (!candidate.getClientRects().length) continue;
      const box = candidate.parentElement.getBoundingClientRect();
      const distance = Math.max(box.top - pointer, pointer - box.bottom, 0);
      if (distance < nearest) {
        nearest = distance;
        list = candidate;
      }
    }
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
    const group = list.parentElement;
    drop(card, group.closest(".category").dataset.category, group.dataset.group === "seen");
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  frame = requestAnimationFrame(scroll);
}

// A dropped card takes the category and the half it lands in, then its place in the order.
function drop(card, category, seen) {
  const paper = papers.find((other) => other.id === card.dataset.id);
  const changes = {};
  if (category !== categoryOf(paper)) changes.category = category || null;
  if (seen !== paper.seen) changes.seen = seen;
  if (Object.keys(changes).length) {
    const before = {category: paper.category, seen: paper.seen};
    if (category) paper.category = category;
    else delete paper.category;
    paper.seen = seen;
    saving = saving.then(async () => {
      try {
        await request("PATCH", `${API}/${encodeURIComponent(paper.id)}`, changes);
        status.textContent = "";
      } catch (error) {
        if (before.category) paper.category = before.category;
        else delete paper.category;
        paper.seen = before.seen;
        layout();
        status.textContent = `Not moved: ${error.message}`;
      }
    });
    settle(card);
  }
  saveOrder(card);
  layout();
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
      await request("PUT", API, {order});
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
    await request("PATCH", `${API}/${encodeURIComponent(paper.id)}`, {seen});
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
  const category = picker.hidden ? "" : picker.value;
  for (const control of form.elements) control.disabled = true;
  status.textContent = "Adding…";
  try {
    const paper = await request("POST", API, category ? {url, category} : {url});
    papers.unshift(paper);
    layout();
    settle(cards.get(paper.id));
    form.elements.url.value = "";
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
    const [list, saved] = await Promise.all([
      fetch("literature/papers.json", {cache: "no-store"}).then((response) => {
        if (!response.ok) throw new Error(response.statusText);
        return response.json();
      }),
      // A server without categories (one not restarted since they came) lists none.
      fetch(CATEGORIES, {cache: "no-store"}).then((response) => (response.ok ? response.json() : {}), () => ({})),
    ]);
    papers = list;
    categories = saved.categories ?? [];
    render();
  } catch {
    status.textContent = "Could not load the literature list.";
  }
})();
