// The pages' text, which the user edits on the page: each element with data-text="<name>" shows
// text/<page>/<name>.md, Markdown with math, rendered as on the Meetings page. The pencil in the
// header switches the page to editing, where clicking a text, or an empty place for one, opens
// the shared editor; serve.py lists a page's texts and saves them, deleting an emptied one.
import {element, render, saveText, showEditor} from "./editor.js";

const PAGE = location.pathname.split("/").pop().replace(/\.html$/, "") || "index";
const EMPTY = '<p class="text-empty">Add text</p>';
const PENCIL = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M13.6 3.6l2.8 2.8L7 15.8l-3.6.8.8-3.6z"/></svg>';
const texts = fetch(`api/text/${PAGE}`, {cache: "no-store"})
  .then((response) => (response.ok ? response.json() : {blocks: {}}))
  .then(({blocks}) => new Map(Object.entries(blocks)))
  .catch(() => new Map());
let editing = false;

// A place for a text: an "intro" sits over a table or figure in the muted intro style, a "note"
// under it, and a "page" text under the page's title.
export function slot(name, kind = "note") {
  const node = element("div", `text-block ${kind}`);
  node.dataset.text = name;
  return node;
}

// Outside editing an empty place shows nothing.
function show(node, text) {
  node.tabIndex = editing ? 0 : -1;
  if (!text.trim() && !editing) {
    node.replaceChildren();
    return;
  }
  const view = element("div", "writeup-view");
  view.innerHTML = text.trim() ? render(text) : EMPTY;
  node.replaceChildren(view);
}

export async function mountText(root = document) {
  const store = await texts;
  for (const node of root.querySelectorAll(".text-block[data-text]:not(.open)")) {
    show(node, store.get(node.dataset.text) ?? "");
  }
}

async function open(node) {
  const store = await texts;
  const name = node.dataset.text;
  const original = store.get(name) ?? "";
  node.classList.add("open");
  showEditor(node, {
    key: node,
    text: original,
    empty: EMPTY,
    save: async ({text}) => {
      if (text === original) return;
      await saveText(`text/${PAGE}/${name}.md`, text);
      store.set(name, text);
    },
    close: () => {
      node.classList.remove("open");
      show(node, store.get(name) ?? "");
    },
  });
}

const toggle = element("button", "text-toggle");
toggle.type = "button";
toggle.title = "Edit the page's text";
toggle.setAttribute("aria-label", "Edit the page's text");
toggle.setAttribute("aria-pressed", "false");
toggle.innerHTML = PENCIL;
toggle.addEventListener("click", () => {
  editing = !editing;
  toggle.setAttribute("aria-pressed", String(editing));
  document.body.classList.toggle("editing-text", editing);
  mountText();
});
document.querySelector("header nav")?.append(toggle);

document.addEventListener("click", (event) => {
  const node = editing && event.target.closest(".text-block[data-text]:not(.open)");
  if (!node) return;
  event.preventDefault();
  open(node);
});
document.addEventListener("keydown", (event) => {
  const node = editing && event.key === "Enter" && event.target.closest?.(".text-block[data-text]:not(.open)");
  if (!node) return;
  event.preventDefault();
  open(node);
});

mountText();
