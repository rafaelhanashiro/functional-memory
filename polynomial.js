// The Polynomial runs page: tables and a chart drawn from data/polynomial.json, which
// data/polynomial.py writes from the Transformer++ result summaries, with places for the user's text.
import {degreeLabel, html, section, table} from "./tables.js";
import {mountText, slot} from "./text.js";

const SVG = "http://www.w3.org/2000/svg";
const COLORS = {taylor: "#2563eb", hermite: "#ea580c", static: "#059669", query_w_decay: "#d97706"};
const MARKERS = {taylor: "circle", hermite: "triangle", static: "diamond", query_w_decay: "square"};

function svg(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

// Bold marks the lowest value as shown, so values that tie at the shown precision are all bold.
function baselines(data, host) {
  const best = Math.min(...data.rows.map((row) => row.perplexity)).toFixed(2);
  section(host, data.title).append(table(["Attention", "PPL"], data.rows.map((row) => [
    {text: row.label}, {text: row.perplexity.toFixed(2), best: row.perplexity.toFixed(2) === best}]), {compact: true}));
}

function initializations(data, host) {
  const best = data.columns.map((column, index) => Math.min(...data.rows.map((row) => row.cells[index].perplexity)).toFixed(2));
  const grid = table(["Initialization", ...data.columns.map((column) => `${column.family}, ${degreeLabel(column.degree)}`)],
    data.rows.map((row) => [{text: row.label}, ...row.cells.map((cell, index) => ({
      text: cell.perplexity.toFixed(2), best: cell.perplexity.toFixed(2) === best[index]}))]), {compact: true});
  section(host, data.title).append(grid);
}

// Bold marks each column's lowest value as shown, in columns with more than one value.
function queryKey(data, host) {
  const shown = (cell) => cell && cell.perplexity.toFixed(4);
  const best = data.columns.map((column, index) => {
    const values = data.rows.map((row) => shown(row.cells[index])).filter(Boolean);
    return values.length > 1 ? Math.min(...values.map(Number)).toFixed(4) : null;
  });
  section(host, data.title).append(table(["Degree", ...data.columns], data.rows.map((row) => [
    {text: degreeLabel(row.degree)},
    ...row.cells.map((cell, index) => cell ? {text: shown(cell), best: shown(cell) === best[index]} : {text: "—"}),
  ]), {compact: true}));
}

function widthDegree(data, host) {
  const values = data.rows.flatMap((row) => row.cells.filter(Boolean).map((cell) => cell.perplexity));
  const best = Math.min(...values).toFixed(4);
  section(host, data.title).append(table(
    ["Key width <var>d</var><sub><var>K</var></sub>", ...data.degrees.map(degreeLabel)],
    data.rows.map((row) => [{text: String(row.width)}, ...row.cells.map((cell) => cell
      ? {text: cell.perplexity.toFixed(4), best: cell.perplexity.toFixed(4) === best} : {text: "—"})]),
  ));
}

function degreeTable(data) {
  const lookup = data.families.map((family) => new Map(family.points.map((point) => [point.degree, point.perplexity])));
  const best = data.degrees.map((degree) => Math.min(...lookup.filter((values) => values.has(degree)).map((values) => values.get(degree))));
  return table(
    ["Attention", ...data.degrees.map(degreeLabel)],
    data.families.map((family, index) => [{text: family.label}, ...data.degrees.map((degree, column) => {
      const value = lookup[index].get(degree);
      return value === undefined ? {text: "—"} : {text: value.toFixed(2), best: value.toFixed(2) === best[column].toFixed(2)};
    })]),
  );
}

function marker(kind, x, y, color, label) {
  const style = {fill: color, stroke: "#fff", "stroke-width": 1.5};
  const shapes = {
    circle: () => svg("circle", {cx: x, cy: y, r: 5.5, ...style}),
    square: () => svg("rect", {x: x - 5, y: y - 5, width: 10, height: 10, ...style}),
    diamond: () => svg("path", {d: `M${x} ${y - 7}L${x + 7} ${y}L${x} ${y + 7}L${x - 7} ${y}Z`, ...style}),
    triangle: () => svg("path", {d: `M${x} ${y - 7}L${x + 6.5} ${y + 5}L${x - 6.5} ${y + 5}Z`, ...style}),
  };
  const shape = shapes[kind]();
  if (label) shape.append(svg("title", {}, label));
  return shape;
}

function degreeChart(data) {
  const width = 960;
  const height = 540;
  const margin = {left: 82, right: 150, top: 24, bottom: 66};
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  // Consecutive degrees sit one slot apart; skipped degrees leave one empty slot.
  const slots = new Map();
  data.degrees.forEach((degree, index) => {
    const previous = data.degrees[index - 1];
    slots.set(degree, index === 0 ? 0 : slots.get(previous) + (degree - previous > 1 ? 2 : 1));
  });
  const slotCount = Math.max(...slots.values()) + 1;
  const x = (degree) => margin.left + (slots.get(degree) + 0.5) * plotWidth / slotCount;
  const values = [...data.families.flatMap((family) => family.points.map((point) => point.perplexity)),
                  ...data.references.map((reference) => reference.perplexity)];
  const low = Math.floor(Math.min(...values)) - 0.2;
  const high = Math.max(...values) + 0.3;
  const y = (value) => margin.top + (high - value) / (high - low) * plotHeight;
  const chart = svg("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": data.title, class: "chart"});

  for (let tick = Math.ceil(low); tick <= high; tick += 1) {
    chart.append(svg("line", {x1: margin.left, x2: margin.left + plotWidth, y1: y(tick), y2: y(tick), class: "grid"}));
    chart.append(svg("text", {x: margin.left - 12, y: y(tick) + 5, class: "tick", "text-anchor": "end"}, String(tick)));
  }
  chart.append(svg("path", {d: `M${margin.left} ${margin.top}V${margin.top + plotHeight}H${margin.left + plotWidth}`, class: "axis"}));
  for (const degree of data.degrees) {
    const label = svg("text", {x: x(degree), y: margin.top + plotHeight + 24, class: "tick", "text-anchor": "middle"});
    label.append(svg("tspan", {"font-style": "italic"}, "m"), ` = ${degree}`);
    chart.append(label);
  }
  chart.append(svg("text", {x: margin.left + plotWidth / 2, y: height - 10, class: "axis-title", "text-anchor": "middle"}, "Polynomial degree"));
  chart.append(svg("text", {x: 0, y: 0, class: "axis-title", "text-anchor": "middle",
    transform: `translate(22 ${margin.top + plotHeight / 2}) rotate(-90)`}, "Perplexity (lower is better)"));

  for (const reference of data.references) {
    const level = y(reference.perplexity);
    chart.append(svg("line", {x1: margin.left, x2: margin.left + plotWidth, y1: level, y2: level,
      class: `reference ${reference.line}`}));
    chart.append(svg("text", {x: margin.left + plotWidth + 10, y: level + 5, class: "reference-label"},
      `${reference.label} ${reference.perplexity.toFixed(2)}`));
  }
  for (const family of data.families) {
    const color = COLORS[family.family];
    family.points.forEach((point, index) => {
      const next = family.points[index + 1];
      if (!next) return;
      const skipped = slots.get(next.degree) - slots.get(point.degree) > 1;
      chart.append(svg("line", {x1: x(point.degree), y1: y(point.perplexity), x2: x(next.degree), y2: y(next.perplexity),
        stroke: color, class: skipped ? "series skipped" : "series"}));
    });
  }
  for (const family of data.families) {
    for (const point of family.points) {
      chart.append(marker(MARKERS[family.family], x(point.degree), y(point.perplexity), COLORS[family.family],
        `${family.legend}, m = ${point.degree}: ${point.perplexity.toFixed(2)}`));
    }
  }
  const legend = svg("g", {transform: `translate(${margin.left + plotWidth - 200} ${margin.top + 18})`});
  data.families.forEach((family, index) => {
    const row = index * 28;
    legend.append(svg("line", {x1: 0, x2: 30, y1: row, y2: row, stroke: COLORS[family.family], class: "series"}));
    legend.append(marker(MARKERS[family.family], 15, row, COLORS[family.family]));
    legend.append(svg("text", {x: 42, y: row + 5, class: "legend"}, family.legend));
  });
  chart.append(legend);
  const wrap = html("div", "chart-wrap");
  wrap.append(chart);
  return wrap;
}

(async () => {
  const main = document.querySelector("main");
  try {
    const response = await fetch("data/polynomial.json", {cache: "no-store"});
    if (!response.ok) throw new Error(response.statusText);
    const data = await response.json();
    baselines(data.baselines, document.querySelector('[data-result="baselines"]'));
    widthDegree(data.width_degree, document.querySelector('[data-result="width-degree"]'));
    section(document.querySelector('[data-result="degree-scaling"]'), data.degree_scaling.title)
      .append(degreeTable(data.degree_scaling), degreeChart(data.degree_scaling));
    initializations(data.initializations, document.querySelector('[data-result="initializations"]'));
    queryKey(data.query_key, document.querySelector('[data-result="query-key"]'));
  } catch {
    main.append(html("p", "empty", "Could not load the results."));
  }
  // Each section's text sits under its heading and its notes under its content, and its heading
  // is renamable (text.js).
  for (const host of document.querySelectorAll(".result[data-result]")) {
    const heading = host.querySelector("h2");
    if (!heading) continue;
    heading.dataset.title = host.dataset.result;
    heading.after(slot(host.dataset.result, "intro"));
    host.append(slot(`${host.dataset.result}-notes`));
  }
  mountText(main);
})();
