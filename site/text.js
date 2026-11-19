// The pages' text and titles, which the user edits on the page. Each element with
// data-text="<name>" shows text/<page>/<name>.md, Markdown with math, rendered as on the Meetings
// page. Each title shows its new name once renamed: the page's heading and the headings marked
// data-title="<key>", and, for other pages, the banners and back links that name them. The pencil
// in the header switches the page to editing, where clicking a text, or an empty place for one,
// opens the shared editor, and clicking a title edits it in place. serve.py lists a page's texts
// and every page's titles, and saves them; an emptied text is deleted, an emptied title restored.
import {element, failure, render, saveText, showEditor} from "./editor.js";

const PAGE = location.pathname.split("/").pop().replace(/\.html$/, "") || "index";
const SITE_NAME = " · Functional Memory";
const EMPTY = '<p class="text-empty">Add text</p>';
const PENCIL = '<svg viewBox="0 0 20 20" aria-hidden="true"><path d="M13.6 3.6l2.8 2.8L7 15.8l-3.6.8.8-3.6z"/></svg>';
const TITLES = "[data-title], .subpage-title";
const store = fetch(`api/text/${PAGE}`, {cache: "no-store"})
  .then((response) => (response.ok ? response.json() : {}))
  .catch(() => ({}))
  .then(({blocks = {}, titles = {}}) => ({
    blocks: new Map(Object.entries(blocks)),
    titles: new Map(Object.entries(titles).map(([page, names]) => [page, new Map(Object.entries(names))])),
  }));
const originals = new WeakMap();  // each title's own HTML, shown again when its new name is cleared
const pageTitle = document.title;
let editing = false;

document.querySelector("h1.page-title")?.setAttribute("data-title", "title");

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

// A title is one line of Markdown, so it may hold math or emphasis; the browser tab and back
// links show it without the marks.
function inline(title) {
  const html = render(title).trim();
  return html.match(/^<p>([\s\S]*)<\/p>$/)?.[1] ?? html;
}
const plain = (title) => title.replace(/[$*_`]/g, "");
const pageOf = (link) => link.getAttribute("href").split("/").pop().replace(/\.html$/, "");
const textOf = (html) => Object.assign(document.createElement("div"), {innerHTML: html}).textContent.trim();

// The title a heading or banner shows: which page it names, and under which key.
function target(node) {
  return node.matches(".subpage-title") ? [pageOf(node.closest(".subpage")), "title"] : [PAGE, node.dataset.title];
}

function original(node) {
  if (!originals.has(node)) originals.set(node, node.innerHTML);
  return originals.get(node);
}

async function showTitles(root) {
  const {titles} = await store;
  const named = (page, key) => titles.get(page)?.get(key);
  for (const node of root.querySelectorAll(TITLES)) {
    if (node.classList.contains("renaming")) continue;
    const own = original(node);
    const title = named(...target(node));
    node.innerHTML = title ? inline(title) : own;
  }
  for (const node of root.querySelectorAll("a.back")) {
    const own = original(node);
    const title = named(pageOf(node), "title");
    if (title) node.textContent = `← ${plain(title)}`;
    else node.innerHTML = own;
  }
  const own = named(PAGE, "title");
  document.title = own ? `${plain(own)}${SITE_NAME}` : pageTitle;
}

export async function mountText(root = document) {
  const {blocks} = await store;
  for (const node of root.querySelectorAll(".text-block[data-text]:not(.open)")) {
    show(node, blocks.get(node.dataset.text) ?? "");
  }
  await showTitles(root);
}

async function open(node) {
  const {blocks} = await store;
  const name = node.dataset.text;
  const before = blocks.get(name) ?? "";
  node.classList.add("open");
  showEditor(node, {
    key: node,
    text: before,
    empty: EMPTY,
    save: async ({text}) => {
      if (text === before) return;
      await saveText(`text/${PAGE}/${name}.md`, text);
      blocks.set(name, text);
    },
    close: () => {
      node.classList.remove("open");
      show(node, blocks.get(name) ?? "");
    },
  });
}

// A title edits in place: Enter or leaving it saves, Escape cancels, and clearing it, or giving
// back its original name, restores the original.
async function rename(node) {
  const {titles} = await store;
  const [page, key] = target(node);
  const before = titles.get(page)?.get(key) ?? "";
  const own = textOf(original(node));
  const input = element("input", "title-input");
  input.value = before || own;
  input.setAttribute("aria-label", "Title");
  node.classList.add("renaming");
  node.replaceChildren(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const typed = input.value.replace(/\s+/g, " ").trim();
    const title = typed === own ? "" : typed;
    if (save && title !== before) {
      try {
        const response = await fetch(`api/titles/${page}`, {
          method: "PUT", headers: {"Content-Type": "application/json"}, body: JSON.stringify({key, title}),
        });
        if (!response.ok) throw new Error(await failure(response));
        titles.set(page, new Map(Object.entries((await response.json()).titles)));
      } catch (error) {
        alert(`The title was not saved: ${error.message}`);
      }
    }
    node.classList.remove("renaming");
    showTitles(document);
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === "Escape") {
      event.preventDefault();
      finish(event.key === "Enter");
    }
  });
  input.addEventListener("blur", () => finish(true));
}

const toggle = element("button", "text-toggle");
toggle.type = "button";
toggle.title = "Edit the page's text and titles";
toggle.setAttribute("aria-label", "Edit the page's text and titles");
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
  if (!editing) return;
  const title = event.target.closest(TITLES);
  if (title) {
    event.preventDefault();  // a banner's title is renamed rather than followed
    if (!title.classList.contains("renaming")) rename(title);
    return;
  }
  const node = event.target.closest(".text-block[data-text]:not(.open)");
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
