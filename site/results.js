// Result pages drawn from a data file that a generator in data/ writes: an element with
// data-source="data/<name>.json" and data-key="<page>" gets that page's sections, each an optional
// heading over its tables, whose cells the generator has already formatted, with places for the
// user's text above and below them (text.js).
import {html, table} from "./tables.js";
import {mountText, slot} from "./text.js";

function render(sections, host) {
  for (const part of sections) {
    const node = html("section", "result");
    if (part.title) node.append(html("h2", "", part.title));
    node.append(slot(part.key, "intro"));
    const tables = part.tables.map((spec) => {
      const wrap = table(spec.head, spec.rows, {compact: Boolean(spec.compact)});
      if (!spec.title) return wrap;
      const item = html("div", "table-item");
      item.append(html("h3", "", spec.title), wrap);
      return item;
    });
    if (part.layout === "pair") {
      const pair = html("div", "table-pair");
      pair.append(...tables);
      node.append(pair);
    } else {
      node.append(...tables);
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
