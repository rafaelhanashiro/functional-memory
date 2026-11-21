// The Queue page: runs not queued yet ("planned") and runs in progress ("active"). Each list holds
// runs and named groups of runs, one level deep, and either list takes new runs and groups (its
// + Run and + Group, or a group's +). A planned run's checkbox moves it to In progress;
// an active run's checkbox closes it out (removes it, after asking), and its arrow sends it back.
// A moved run lands under the group of the same name on the other side, made if missing. A group
// disappears once its last run moves to the other side (or, in In progress, is closed out); in
// Not queued, one emptied by deleting or dragging its runs stays. Planned runs and groups reorder
// by their grip (drag, or the arrow keys), and runs drag in and out of groups. Every change saves
// both lists through serve.py (api/queue), which refuses a save from a page that is out of date.
import {element, failure, render} from "./editor.js";

const API = "api/queue";
const GRIP = '<svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="7.5" cy="5" r="1.4"/><circle cx="12.5" cy="5" r="1.4"/><circle cx="7.5" cy="10" r="1.4"/><circle cx="12.5" cy="10" r="1.4"/><circle cx="7.5" cy="15" r="1.4"/><circle cx="12.5" cy="15" r="1.4"/></svg>';
const CHECK = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 10.5l3 3 6-7"/></svg>';
const BACK = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M16 10H4M9 5l-5 5 5 5"/></svg>';
const PLUS = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M10 4v12M4 10h12"/></svg>';
const CROSS = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5.5 5.5l9 9M14.5 5.5l-9 9"/></svg>';
const MAX_TEXT = 500;
const MAX_NAME = 200;

let state = {revision: 0, lists: {planned: [], active: []}};
let saving = Promise.resolve();

const newId = () => crypto.randomUUID?.() ?? [...crypto.getRandomValues(new Uint8Array(12))]
  .map((byte) => byte.toString(16).padStart(2, "0")).join("");
const plain = (text) => text.replace(/[$*_`]/g, "");
const runsIn = (items) => items.reduce((count, item) => count + (item.type === "group" ? item.runs.length : 1), 0);

// A run's text is one line of Markdown, so it may hold math or emphasis.
function inline(text) {
  const html = render(text).trim();
  return html.match(/^<p>([\s\S]*)<\/p>$/)?.[1] ?? html;
}

// The model: where an id sits in a list, and moving runs between the lists.

function find(list, id) {
  const items = state.lists[list];
  for (const [index, item] of items.entries()) {
    if (item.id === id) return {container: items, index, item, group: null};
    if (item.type !== "group") continue;
    const at = item.runs.findIndex((run) => run.id === id);
    if (at >= 0) return {container: item.runs, index: at, item: item.runs[at], group: item};
  }
  return null;
}

function remove(list, id) {
  const found = find(list, id);
  found.container.splice(found.index, 1);
  if (list === "active" && found.group && !found.group.runs.length) remove(list, found.group.id);
  return found;
}

function transfer(id, from, to) {
  const {item, group} = remove(from, id);
  if (group && !group.runs.length && find(from, group.id)) remove(from, group.id);  // its last run left
  const items = state.lists[to];
  if (!group) {
    items.push(item);
    return;
  }
  let target = items.find((other) => other.type === "group" && other.name === group.name);
  if (!target) {
    target = {id: newId(), type: "group", name: group.name, runs: []};
    items.push(target);
  }
  target.runs.push(item);
}

// Saving: each change saves a copy of both lists, in order, against the revision the last save
// returned. A refused or failed save reloads the lists from the server.

function commit() {
  draw();
  const lists = structuredClone(state.lists);
  saving = saving.then(async () => {
    try {
      const response = await fetch(API, {
        method: "PUT", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({revision: state.revision, lists}),
      });
      if (!response.ok) throw new Error(await failure(response));
      state.revision = (await response.json()).revision;
    } catch (error) {
      alert(`The queue was not saved: ${error.message}`);
      await load();
    }
  });
}

async function load() {
  const response = await fetch(API, {cache: "no-store"});
  if (!response.ok) throw new Error(await failure(response));
  state = await response.json();
  draw();
}

// Drawing: a row per run and per group heading, with the group's runs under it.

function icon(kind, label, svg, onClick) {
  const node = element("button", `queue-icon ${kind}`);
  node.type = "button";
  node.title = label;
  node.setAttribute("aria-label", label);
  node.innerHTML = svg;
  node.addEventListener("click", onClick);
  return node;
}

function checkbox(label, onCheck, disabled = false) {
  const node = element("button", "queue-check");
  node.type = "button";
  node.title = label;
  node.setAttribute("aria-label", label);
  node.innerHTML = CHECK;
  node.disabled = disabled;
  node.addEventListener("click", () => {
    node.classList.add("checked");
    // Let the tick show before the row moves or the confirmation opens.
    setTimeout(() => {
      if (onCheck() === false) node.classList.remove("checked");
    }, 120);
  });
  return node;
}

function grip(node, label) {
  const handle = element("button", "drag-handle");
  handle.type = "button";
  handle.innerHTML = GRIP;
  handle.title = "Drag to reorder";
  handle.setAttribute("aria-label", `Reorder ${label}: drag, or press the up and down arrow keys`);
  handle.addEventListener("pointerdown", (event) => drag(event, node, handle));
  handle.addEventListener("keydown", (event) => nudge(event, node));
  return handle;
}

function label(className, text, empty, onEdit) {
  const node = element("span", className);
  if (text) node.innerHTML = inline(text);
  else node.append(element("span", "queue-placeholder", empty));
  node.addEventListener("click", (event) => {
    if (!event.target.closest("a")) onEdit(node);
  });
  return node;
}

function runNode(list, run) {
  const node = element("li", "queue-run");
  node.dataset.id = run.id;
  const row = element("div", "queue-row");
  if (list === "planned") row.append(grip(node, "this run"));
  row.append(
    list === "planned" ? checkbox("Queued: move to In progress", () => { transfer(run.id, "planned", "active"); commit(); })
      : checkbox("Closed out: remove from the list", () => closeOut(run.id)),
    label("queue-text", run.text, "Describe the run", (span) => edit(span, run, "text")),
  );
  const actions = element("span", "queue-actions");
  if (list === "active") {
    actions.append(icon("back", "Back to Not queued", BACK, () => { transfer(run.id, "active", "planned"); commit(); }));
  } else {
    actions.append(icon("delete", "Delete this run", CROSS, () => {
      if (!run.text || confirm(`Delete "${plain(run.text)}"?`)) {
        remove("planned", run.id);
        commit();
      }
    }));
  }
  row.append(actions);
  node.append(row);
  return node;
}

function groupNode(list, group) {
  const node = element("li", "queue-group");
  node.dataset.id = group.id;
  const row = element("div", "queue-row queue-group-head");
  if (list === "planned") row.append(grip(node, "this group"));
  const count = element("span", "queue-group-count", String(group.runs.length));
  row.append(
    list === "planned"
      ? checkbox("Queued: move all its runs to In progress", () => moveGroup(group, "planned", "active"), !group.runs.length)
      : checkbox("Closed out: remove the group and its runs", () => closeOutGroup(group)),
    label("queue-name", group.name, "Name the group", (span) => edit(span, group, "name")),
    count,
  );
  const actions = element("span", "queue-actions");
  if (list === "active") {
    actions.append(icon("add", "Add a run to this group", PLUS, () => add(group.runs)));
    if (group.runs.length) actions.append(icon("back", "Send all its runs back to Not queued", BACK, () => moveGroup(group, "active", "planned")));
  } else {
    actions.append(
      icon("add", "Add a run to this group", PLUS, () => add(group.runs)),
      icon("delete", "Delete this group", CROSS, () => {
        const runs = group.runs.length ? ` and its ${group.runs.length} run${group.runs.length > 1 ? "s" : ""}` : "";
        if (!group.name && !runs || confirm(`Delete the group "${plain(group.name)}"${runs}?`)) {
          remove("planned", group.id);
          commit();
        }
      }),
    );
  }
  row.append(actions);
  const runs = element("ul", "queue-items queue-group-runs");
  runs.append(...group.runs.map((run) => runNode(list, run)));
  node.append(row, runs);
  return node;
}

function draw() {
  for (const list of ["planned", "active"]) {
    const panel = document.querySelector(`[data-list="${list}"]`);
    const items = state.lists[list];
    const runs = runsIn(items);
    panel.querySelector(".queue-count").textContent = runs ? String(runs) : "";
    panel.querySelector(".queue-items").replaceChildren(
      ...items.map((item) => item.type === "group" ? groupNode(list, item) : runNode(list, item)));
    panel.querySelector(".queue-empty").hidden = items.length > 0;
  }
}

// Actions on runs and groups.

function closeOut(id) {
  const {item} = find("active", id);
  if (!confirm(`Close out "${plain(item.text)}"? It leaves the list.`)) return false;
  remove("active", id);
  commit();
}

function closeOutGroup(group) {
  const count = group.runs.length;
  if (!confirm(`Close out "${plain(group.name)}" and its ${count} run${count === 1 ? "" : "s"}? They leave the list.`)) return false;
  remove("active", group.id);
  commit();
}

function moveGroup(group, from, to) {
  for (const run of [...group.runs]) transfer(run.id, from, to);
  commit();
}

// A new run or group goes to the top of its list, or of its group, and opens for its text.
function add(container, group = false) {
  const item = group ? {id: newId(), type: "group", name: "", runs: []} : {id: newId(), type: "run", text: ""};
  container.unshift(item);
  draw();
  const span = document.querySelector(`[data-id="${item.id}"] > .queue-row > ${group ? ".queue-name" : ".queue-text"}`);
  edit(span, item, group ? "name" : "text", true);
}

// Text edits in place: Enter or leaving the field saves, Escape cancels. A new item left empty
// disappears; emptying an existing one keeps its old text.
function edit(span, item, field, fresh = false) {
  if (span.querySelector("input")) return;
  const input = element("input", "queue-input");
  input.value = item[field];
  input.maxLength = field === "name" ? MAX_NAME : MAX_TEXT;
  input.placeholder = field === "name" ? "Group name" : "Run";
  span.replaceChildren(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = (save) => {
    if (finished) return;
    finished = true;
    const value = input.value.replace(/\s+/g, " ").trim();
    if (save && value && value !== item[field]) {
      item[field] = value;
      commit();
    } else if (fresh && !item[field]) {
      const found = find("planned", item.id) ?? find("active", item.id);
      found.container.splice(found.index, 1);  // only the new item, never the group it was added to
      draw();
    } else {
      draw();
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

// Reordering Not queued: the grip lifts its row and a dashed slot follows the pointer. A group
// moves among the top-level items; a run also moves into a group, under its heading, or out of
// one. The drop places the item where the slot is and saves.

function drag(event, node, handle) {
  if (event.button !== 0) return;
  event.preventDefault();
  const top = document.querySelector('[data-list="planned"] > .queue-items');
  const isGroup = node.classList.contains("queue-group");
  const start = node.getBoundingClientRect();
  const offset = event.clientY - start.top;
  const slot = element("li", "queue-slot");
  slot.style.height = `${start.height}px`;
  node.parentElement.insertBefore(slot, node);
  Object.assign(node.style, {
    position: "fixed", left: `${start.left}px`, top: `${start.top}px`, width: `${start.width}px`, zIndex: "20",
  });
  node.classList.add("dragging");
  document.body.classList.add("reordering");
  handle.setPointerCapture(event.pointerId);
  let pointer = event.clientY;
  let frame = 0;

  const target = () => {
    if (isGroup) return top;
    for (const group of top.children) {
      if (group === node || !group.classList.contains("queue-group")) continue;
      const head = group.firstElementChild.getBoundingClientRect();
      if (pointer > head.top + head.height / 2 && pointer < group.getBoundingClientRect().bottom) {
        return group.querySelector(".queue-group-runs");
      }
    }
    return top;
  };
  const place = () => {
    node.style.top = `${pointer - offset}px`;
    const list = target();
    let before = null;
    for (const sibling of list.children) {
      if (sibling === node || sibling === slot) continue;
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
    const list = slot.parentElement;
    const index = [...list.children].filter((child) => child !== node).indexOf(slot);
    slot.remove();
    node.removeAttribute("style");
    node.classList.remove("dragging");
    document.body.classList.remove("reordering");
    const {item} = remove("planned", node.dataset.id);
    const container = list === top ? state.lists.planned : find("planned", list.parentElement.dataset.id).item.runs;
    container.splice(index, 0, item);
    commit();
  };
  handle.addEventListener("pointermove", move);
  handle.addEventListener("pointerup", end);
  handle.addEventListener("pointercancel", end);
  frame = requestAnimationFrame(scroll);
}

function nudge(event, node) {
  const up = event.key === "ArrowUp";
  if (!up && event.key !== "ArrowDown") return;
  event.preventDefault();
  const {container, index} = find("planned", node.dataset.id);
  const next = index + (up ? -1 : 1);
  if (next < 0 || next >= container.length) return;
  container.splice(next, 0, ...container.splice(index, 1));
  commit();
  document.querySelector(`[data-id="${node.dataset.id}"] > .queue-row > .drag-handle`)?.focus();
}

for (const panel of document.querySelectorAll(".queue-panel")) {
  const list = panel.dataset.list;
  panel.querySelector('[data-action="add-run"]').addEventListener("click", () => add(state.lists[list]));
  panel.querySelector('[data-action="add-group"]').addEventListener("click", () => add(state.lists[list], true));
}
load().catch(() => {
  for (const panel of document.querySelectorAll(".queue-panel")) {
    panel.querySelector(".queue-empty").textContent = "Could not load the queue.";
    panel.querySelector(".queue-empty").hidden = false;
  }
});
