// Shared by the result pages: helpers that build section headings and results tables.

export function html(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.innerHTML = content;
  return node;
}

export const degreeLabel = (degree) => `<var>m</var> = ${degree}`;

export function section(host, title) {
  host.replaceChildren(html("h2", "", title));
  return host;
}

export const intro = (text) => html("p", "intro", text);

// A results table. Each body row is a list of cells, or {cells, className}; a cell is
// {text, best, muted, rowspan, label}, where best bolds it, muted grays it, and label marks a
// row-group name.
export function table(head, body, {compact = false, grouped = false} = {}) {
  const wrap = html("div", compact ? "table-wrap compact" : "table-wrap");
  const element = html("table", grouped ? "results-table grouped" : "results-table");
  const header = html("tr");
  for (const cell of head) header.append(html("th", "", cell));
  element.append(html("thead"), html("tbody"));
  element.tHead.append(header);
  for (const entry of body) {
    const {cells, className} = Array.isArray(entry) ? {cells: entry} : entry;
    const row = html("tr", className);
    for (const {text, best, muted, rowspan, label} of cells) {
      const cell = html("td", [best && "best", muted && "muted", label && "group-label"].filter(Boolean).join(" "), text);
      if (rowspan) cell.rowSpan = rowspan;
      row.append(cell);
    }
    element.tBodies[0].append(row);
  }
  wrap.append(element);
  return wrap;
}
