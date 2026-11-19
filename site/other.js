// The Other runs page: tables drawn from data/other.json, which data/other.py writes from the
// runs' records and the Transformer++ result summaries.
import {degreeLabel, html, intro, section, table} from "./tables.js";

const value = (cell) => cell ? {text: cell.perplexity.toFixed(4)} : {text: "—"};

function powers(data, host) {
  section(host, data.title).append(intro(data.intro), table(
    ["Attention", ...data.columns.map((column) => `${column.label} ${degreeLabel(column.degree)}`)],
    data.rows.map((row) => [{text: row.label}, ...row.cells.map(value)]),
  ));
}

function gates(data, host) {
  section(host, data.title).append(intro(data.intro), table(
    ["Degree", "Gate", "PPL"],
    data.rows.map((row) => [{text: degreeLabel(row.degree)}, {text: row.gate}, value(row)]),
    {compact: true},
  ));
}

// Each variant names a group of rows, one per degree; alternate groups are shaded.
function inputNorm(data, host) {
  const body = data.groups.flatMap((group, index) => group.rows.map((row, position) => ({
    className: index % 2 ? "shade" : "",
    cells: [...(position ? [] : [{text: group.label, rowspan: group.rows.length, label: true}]),
            {text: degreeLabel(row.degree)}, ...row.cells.map(value)],
  })));
  section(host, data.title).append(intro(data.intro),
    table(["Variant", "Total degree", ...data.columns], body, {grouped: true}));
}

function zeroConstant(data, host) {
  section(host, data.title).append(intro(data.intro), table(
    ["Variant", "PPL"], data.rows.map((row) => [{text: row.label}, value(row)]), {compact: true},
  ));
}

// Subsections of one heading, each a table whose cells are text or a run's perplexity.
function parts(data, host) {
  section(host, data.title);
  for (const part of data.parts) {
    const rows = part.rows.map((row) => row.map((entry) => typeof entry === "string" ? {text: entry} : value(entry)));
    host.append(html("h3", "", part.title), intro(part.intro), table(part.head, rows, {compact: Boolean(part.compact)}));
  }
}

(async () => {
  const main = document.querySelector("main");
  try {
    const response = await fetch("data/other.json", {cache: "no-store"});
    if (!response.ok) throw new Error(response.statusText);
    const data = await response.json();
    const host = (name) => document.querySelector(`[data-result="${name}"]`);
    powers(data.powers, host("powers"));
    gates(data.gates, host("gates"));
    inputNorm(data.input_norm, host("input-norm"));
    zeroConstant(data.zero_constant, host("zero-constant"));
    parts(data.decay_and_lr, host("decay-and-lr"));
  } catch {
    main.append(Object.assign(document.createElement("p"), {className: "empty", textContent: "Could not load the results."}));
  }
})();
