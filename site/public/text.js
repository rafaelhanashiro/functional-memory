// Load the text and titles included in the public results snapshot.
import {element, render} from "./markdown.js";

const PAGE = location.pathname.split("/").pop().replace(/\.html$/, "") || "index";
const SITE_NAME = " · Functional Memory";
const TITLES = "[data-title], .subpage-title";
const store = fetch("data/page-text.json", {cache: "no-store"})
  .then((response) => (response.ok ? response.json() : {}))
  .catch(() => ({}))
  .then(({pages = {}, titles = {}}) => ({
    blocks: new Map(Object.entries(pages[PAGE] ?? {})),
    titles: new Map(Object.entries(titles).map(([page, names]) => [page, new Map(Object.entries(names))])),
  }));
const originals = new WeakMap();  // each title's own HTML, shown again when its new name is cleared
const pageTitle = document.title;

document.querySelector("h1.page-title")?.setAttribute("data-title", "title");

// A place for a text: an "intro" sits over a table or figure in the muted intro style, a "note"
// under it, and a "page-text" one under the page's title.
export function slot(name, kind = "note") {
  const node = element("div", `text-block ${kind}`);
  node.dataset.text = name;
  return node;
}

function show(node, text) {
  if (!text.trim()) {
    node.replaceChildren();
    return;
  }
  const view = element("div", "writeup-view");
  view.innerHTML = render(text);
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
  for (const node of root.querySelectorAll(".text-block[data-text]")) {
    show(node, blocks.get(node.dataset.text) ?? "");
  }
  await showTitles(root);
}

mountText();
