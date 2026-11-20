// The literature list in the public results snapshot, read-only: literature/papers.json in its
// stored order under the user's categories (literature/categories.json, in their order), each split
// into seen and not seen yet; resources under no category come last, under no heading while there
// are no categories. A card opens to its write-up (literature/<id>.md). The chevron before a
// category's name hides it down to its heading, in this browser only.
import {element, render} from "./markdown.js";

const EMPTY = '<p class="empty">No write-up yet.</p>';
const CHEVRON = '<svg class="chevron" viewBox="0 0 20 20" aria-hidden="true"><path d="M5 7.5l5 5 5-5"/></svg>';
const HALVES = [["seen", "Seen"], ["unseen", "Not seen yet"]];
const FOLDED = "literature-hidden";  // the categories this browser hides
const container = document.querySelector(".categories");
const status = document.querySelector(".literature-status");
const folded = new Set((() => {
  try { return JSON.parse(localStorage.getItem(FOLDED) ?? "[]"); } catch { return []; }
})());

function authorsLabel(authors) {
  const surname = (name) => name.trim().split(/\s+/).pop();
  if (authors.length === 0) return "";
  if (authors.length === 1) return surname(authors[0]);
  if (authors.length === 2) return `${surname(authors[0])} and ${surname(authors[1])}`;
  return `${surname(authors[0])} et al.`;
}

function createCard(paper) {
  const card = element("details", `paper ${paper.seen ? "seen" : "unseen"}`);
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
  summary.append(head);
  summary.insertAdjacentHTML("beforeend", CHEVRON);
  const writeup = element("div", "writeup");
  card.append(summary, writeup);
  card.addEventListener("toggle", async () => {
    if (!card.open || writeup.childElementCount) return;
    try {
      const response = await fetch(`literature/${paper.id}.md`, {cache: "no-store"});
      if (!response.ok) throw new Error(response.statusText);
      const text = await response.text();
      const view = element("div", "writeup-view");
      view.innerHTML = text.trim() ? render(text) : EMPTY;
      writeup.replaceChildren(view);
    } catch {
      writeup.innerHTML = '<p class="empty">Could not load the write-up.</p>';
    }
  });
  return card;
}

// A hidden category folds to its heading, which keeps its count; the choice stays in this browser.
function showFold(section, id) {
  const hidden = folded.has(id);
  const button = section.querySelector(".category-icon.fold");
  const label = hidden ? "Show this category" : "Hide this category";
  section.classList.toggle("folded", hidden);
  button.title = label;
  button.setAttribute("aria-label", label);
  button.setAttribute("aria-expanded", String(!hidden));
}

function createSection(id, name, papers) {
  const section = element("section", id ? "category" : "category uncategorized");
  const head = element("div", "category-head");
  const fold = element("button", "category-icon fold");
  fold.type = "button";
  fold.innerHTML = CHEVRON;
  fold.addEventListener("click", () => {
    if (folded.has(id)) folded.delete(id);
    else folded.add(id);
    try { localStorage.setItem(FOLDED, JSON.stringify([...folded])); } catch { /* hiding is only a convenience */ }
    showFold(section, id);
  });
  head.append(fold, element("h2", "category-name", name), element("span", "category-count", papers.length ? String(papers.length) : ""));
  section.append(head);
  for (const [half, label] of HALVES) {
    const cards = papers.filter((paper) => (half === "seen") === Boolean(paper.seen)).map(createCard);
    const group = element("div", "resource-group");
    group.hidden = !id && cards.length === 0;
    const list = element("div", "papers");
    list.append(...cards);
    group.append(element("h3", "", label), list);
    section.append(group);
  }
  showFold(section, id);
  return section;
}

(async () => {
  try {
    const [papers, categories] = await Promise.all(["literature/papers.json", "literature/categories.json"].map(
      (path) => fetch(path, {cache: "no-store"}).then((response) => {
        if (!response.ok) throw new Error(response.statusText);
        return response.json();
      })));
    const known = new Set(categories.map((category) => category.id));
    const under = (id) => papers.filter((paper) => (known.has(paper.category) ? paper.category : "") === id);
    const sections = categories.map((category) => createSection(category.id, category.name, under(category.id)));
    const rest = under("");
    if (rest.length) sections.push(createSection("", "No category", rest));
    container.replaceChildren(...sections);
    document.body.classList.toggle("has-categories", categories.length > 0);
  } catch {
    status.textContent = "Could not load the literature list.";
  }
})();
