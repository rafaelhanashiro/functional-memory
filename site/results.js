// Result pages drawn from a data file that a generator in data/ writes: an element with
// data-source="data/<name>.json" and data-key="<page>" gets that page's sections, each an optional
// heading over its tables and figures, whose cells the generator has already formatted (rows may be grouped
// under a label, and a table may carry a caption), with places for the user's text above and below
// them and the heading renamable (text.js).
import {html, table} from "./tables.js";
import {mountText, slot} from "./text.js";

function render(sections, host) {
  for (const part of sections) {
    const node = html("section", "result");
    if (part.title) {
      const heading = html("h2", "", part.title);
      heading.dataset.title = part.key;
      node.append(heading);
    }
    node.append(slot(part.key, "intro"));
    const tables = part.tables.map((spec) => {
      const wrap = table(spec.head, spec.rows, {compact: Boolean(spec.compact), grouped: Boolean(spec.grouped)});
      if (!spec.title && !spec.caption) return wrap;
      const item = html("div", "table-item");
      if (spec.title) item.append(html("h3", "", spec.title));
      item.append(wrap);
      if (spec.caption) item.append(html("p", "table-caption", spec.caption));
      return item;
    });
    if (part.layout === "pair") {
      const pair = html("div", "table-pair");
      pair.append(...tables);
      node.append(pair);
    } else {
      node.append(...tables);
    }
    for (const {src, alt} of part.figures ?? []) {
      const link = html("a", "figure");
      link.href = src;
      link.append(Object.assign(document.createElement("img"), {src, alt, loading: "lazy"}));
      node.append(link);
    }
    node.append(slot(`${part.key}-notes`));
    host.append(node);
  }
}

for (const host of document.querySelectorAll("[data-source]")) {
  fetch(host.dataset.source, {cache: "no-store"})
    .then((response) => {
      if (!response.ok) throw new Error(response.statusText);
      return response.json();
    })
    .then((data) => {
      render(data[host.dataset.key], host);
      mountText(host);
    })
    .catch(() => host.append(html("p", "empty", "Could not load the results.")));
}
