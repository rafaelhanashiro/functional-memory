// Shared by the pages: Markdown rendering with math ($...$, $$...$$, \(...\), \[...\]; code
// spans and blocks are matched first so their dollar signs stay literal), and the editor:
// Markdown beside a live preview, with images pasted or dropped into the text uploaded
// through serve.py.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;
const OPEN = String.fromCharCode(0xe000);  // private-use characters mark math in the Markdown
const CLOSE = String.fromCharCode(0xe001);
const PLACEHOLDER = new RegExp(`${OPEN}(\\d+)${CLOSE}`, "g");
const MARKERS = /^(?:[ \t]*(?:>|(?:[-*+]|\d{1,9}[.)])(?=[ \t])))*/;  // quote and list markers opening a line
const QUOTE = /[ \t]*>[ \t]?/y;

// Sheets with unsaved edits: leaving the page asks first.
export const dirty = new Set();
window.addEventListener("beforeunload", (event) => {
  if (dirty.size) event.preventDefault();
});

// Callouts, as in Obsidian: a quote whose first line is [!type], then optionally - (folded) or
// + (foldable, open) and a title, as in "> [!warning]- Careful". Without a title it is just the
// coloured box (a foldable one keeps its type as the title, to click on). style.css colours each
// type, and takes a colour's own name too, as in "> [!green]"; a type it doesn't know is blue, like
// a note.
const CALLOUT = /^<p>\[!([\w-]+)\]([+-]?)[ \t]*(.*?)[ \t]*(\n|<\/p>\n?)/;
marked.use({renderer: {
  blockquote(quote) {
    const match = quote.match(CALLOUT);
    if (!match) return false;
    const [head, name, fold, title, end] = match;
    const type = name.toLowerCase();
    const body = (end === "\n" ? "<p>" : "") + quote.slice(head.length);
    const [box, top] = fold ? ["details", "summary"] : ["div", "div"];
    const heading = title || (fold ? type[0].toUpperCase() + type.slice(1) : "");
    return `<${box} class="callout" data-callout="${type}"${fold === "+" ? " open" : ""}>\n`
      + (heading ? `<${top} class="callout-title">${heading}</${top}>\n` : "")
      + (body.trim() ? `<div class="callout-content">\n${body}</div>\n` : "")
      + `</${box}>\n`;
  },
}});

// Folded callouts open while the page prints, so a PDF of it shows them whole.
let unfolded = [];
window.addEventListener("beforeprint", () => {
  unfolded = [...document.querySelectorAll("details.callout:not([open])")];
  for (const callout of unfolded) callout.open = true;
});
window.addEventListener("afterprint", () => {
  for (const callout of unfolded) callout.open = false;
  unfolded = [];
});

export function render(markdown) {
  const math = [];
  const source = markdown.replace(MATH, (match, code, display, bracket, paren, inline, offset) => {
    if (code !== undefined) return code;
    const block = display !== undefined || bracket !== undefined;
    const tex = unquote(display ?? bracket ?? paren ?? inline, markdown, offset);
    math.push(katex.renderToString(tex, {displayMode: block, throwOnError: false}));
    return `${OPEN}${math.length - 1}${CLOSE}`;
  });
  return marked.parse(source).replace(PLACEHOLDER, (match, index) => math[index]);
}

// Math that starts in a quote may repeat the quote's > markers on its later lines, and those
// markers are Markdown, not TeX. Each comes off where the math's first line has it, give or
// take the three spaces of indent Markdown allows; lines without the markers stay as they are.
function unquote(tex, markdown, offset) {
  const line = markdown.slice(offset && markdown.lastIndexOf("\n", offset - 1) + 1, offset);
  const quotes = line.match(MARKERS)[0].split(">").slice(0, -1);  // what comes before each >
  if (!quotes.length) return tex;
  return tex.split("\n").map((text, index) => {
    if (!index) return text;
    let start = 0;
    for (const before of quotes) {
      QUOTE.lastIndex = start;
      const marker = QUOTE.exec(text);
      if (!marker || marker[0].indexOf(">") > before.length + 3) break;
      start = QUOTE.lastIndex;
    }
    return text.slice(start);
  }).join("\n");
}

export function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text) node.textContent = text;
  return node;
}

export function button(label, kind, onClick) {
  const node = element("button", `button ${kind}`, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

export async function failure(response) {
  try {
    return (await response.json()).error;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function saveText(path, text) {
  const response = await fetch(path, {
    method: "PUT", body: text, headers: {"Content-Type": "text/markdown; charset=utf-8"},
  });
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
}

async function uploadImage(file) {
  const response = await fetch("api/media", {method: "POST", headers: {"Content-Type": file.type}, body: file});
  if (!response.ok) throw new Error(await failure(response));
  return (await response.json()).path;
}

// The rendered text with an "Edit" button, and to its left any other buttons given.
export function showView(host, text, {empty, onEdit, actions = []}) {
  const view = element("div", "writeup-view");
  view.innerHTML = text.trim() ? render(text) : empty;
  const bar = element("div", "view-actions");
  bar.append(...actions, button("Edit", "edit", onEdit));
  host.replaceChildren(bar, view);
}

// Markdown keys in the editor's text box, as in Obsidian: Tab and Shift+Tab move list items a
// level in or out of their list, with the lines under them, and indent or outdent other lines;
// Enter continues a list and ends it on an empty item; Cmd+B, I and K (Ctrl off a Mac) toggle
// bold, italics and a link; and typing a bracket, $ or ` with text selected wraps it. Edits go
// through the browser's undo history. Indents count in columns, a tab reaching the next
// multiple of four as in Markdown, so lists indented with tabs or spaces both work.
const LIST_ITEM = /^([ \t]*)([-*+]|(\d+)([.)]))( +)(\[[ xX]\] +)?/;
const INDENT = "    ";  // four spaces nest a list item under any marker (-, 1., 10.)
const TAB = INDENT.length;  // columns to a tab stop, and the nesting step while nothing is nested
const FENCE = /^[ \t]*(```|~~~)/;
// Unindented lines that start a new block, so end a list even right after one of its lines.
const BLOCK = /^(#{1,6}(\s|$)|>|```|~~~|([-*_])(\s*\3){2,}\s*$)/;
const PAIRS = {"(": ")", "[": "]", "{": "}", "$": "$", "`": "`"};
// On a Mac, Ctrl+B and Ctrl+K already edit text (back a character, delete to the end of the
// line), so the shortcuts take Cmd there.
const MAC = /mac|iphone|ipad/i.test(navigator.userAgentData?.platform || navigator.platform);

function edit(input, start, end, text, selectionStart, selectionEnd = selectionStart) {
  input.setSelectionRange(start, end);
  const done = text ? document.execCommand("insertText", false, text)
    : start === end || document.execCommand("delete");
  if (!done) {
    input.setRangeText(text, start, end, "end");
    input.dispatchEvent(new Event("input", {bubbles: true}));
  }
  input.setSelectionRange(selectionStart, selectionEnd);
}

function selectedLines(input) {
  const {value, selectionStart: start} = input;
  let end = input.selectionEnd;
  if (end > start && value[end - 1] === "\n") end -= 1;  // whole lines selected with their newline
  const first = start && value.lastIndexOf("\n", start - 1) + 1;
  const last = value.indexOf("\n", end);
  return [first, last === -1 ? value.length : last];
}

// The column that leading whitespace reaches, a tab advancing to the next multiple of four.
function columns(space) {
  let column = 0;
  for (const char of space) column = char === "\t" ? column - column % TAB + TAB : column + 1;
  return column;
}

// Leading whitespace changed to reach `column`: with `tabs` (the text's lists indent with tabs)
// tabs then spaces, otherwise the tabs and spaces of `space` that still fit, then spaces.
function reach(space, column, tabs) {
  if (tabs) return "\t".repeat(Math.floor(column / TAB)) + " ".repeat(column % TAB);
  let kept = "";
  for (const char of space) {
    if (columns(kept + char) > column) break;
    kept += char;
  }
  return kept + " ".repeat(column - columns(kept));
}

// The lines of `text`, each with its leading whitespace and indent and, outside fenced code, its
// list item match and content column (where its text, and an item nested under it, starts).
function parse(text) {
  let fenced = false;
  return text.split("\n").map((line) => {
    const space = line.match(/^[ \t]*/)[0];
    const fence = FENCE.test(line);
    const match = fenced || fence ? null : line.match(LIST_ITEM);
    if (fence) fenced = !fenced;
    const margin = columns(space);
    return {line, space, margin, match, content: match && margin + match[2].length + match[5].length};
  });
}

// Whether the text's lists indent with tabs, as Markdown pasted from Obsidian does.
function tabbed(rows) {
  return rows.some(({match, space}) => match && space.includes("\t"));
}

// Whether row `row` is unindented text that ends the list above it: a new block, or a paragraph
// after a blank line. Unindented text right after a list line continues that item's text.
function ends(rows, row) {
  const {line, match, margin} = rows[row];
  if (match || margin || !line.trim()) return false;
  return BLOCK.test(line) || !row || !rows[row - 1].line.trim();
}

// A level of nesting, as the text's lists already nest: the smallest step from a list item to
// its parent (two spaces in lists typed that way), or four (TAB) while nothing is nested. An
// item indented so far that Markdown reads it as text doesn't count.
function nesting(rows) {
  let step;
  const open = [];  // [indent, content column] of the items a later line can nest under
  rows.forEach(({match, margin, content}, row) => {
    if (ends(rows, row)) open.length = 0;
    if (!match) return;
    while (open.length && margin < open.at(-1)[1]) open.pop();
    if (margin >= (open.at(-1)?.[1] ?? 0) + TAB) return;
    if (open.length) step = Math.min(step ?? Infinity, margin - open.at(-1)[0]);
    open.push([margin, content]);
  });
  return step ?? TAB;
}

// The list around the item indented by `depth` on row `row`, from the rows above it: its parent
// (its row, indent and content column), its previous sibling's content column, and the indent
// of the items nested under that sibling.
function relatives(rows, row, depth) {
  const found = {};
  for (let above = row - 1; above >= 0 && !ends(rows, above); above -= 1) {
    const {match, margin, content} = rows[above];
    if (!match) continue;
    if (margin < depth) return {...found, parent: {row: above, margin, content}};
    if (margin === depth) {
      found.sibling ??= content;
      if (!depth) break;  // a top-level item has no parent to find
    } else if (found.sibling === undefined) found.children = Math.min(found.children ?? margin, margin);
  }
  return found;
}

// Where Tab moves the list item indented by `depth` on row `row`: in under its previous sibling
// (or under the item above, if not nested in it yet), level with the items nested there, else a
// nesting step in, at a column where Markdown nests it. With nothing above to nest under, it
// stays put.
function level(rows, row, depth) {
  const {parent, sibling, children} = relatives(rows, row, depth);
  const under = sibling ?? (parent && depth < parent.content ? parent.content : undefined);
  if (under === undefined) return depth;
  // An item nests at the content column of the item above and up to three columns past it.
  const target = children < under + TAB ? children : depth + nesting(rows);
  return target >= under && target < under + TAB ? target : under;
}

// Where Shift+Tab, or Enter on an empty item, moves a nested list item, given its parent from
// relatives(): out to the parent's indent.
function outer(parent, depth) {
  return parent ? parent.margin : Math.max(0, depth - TAB);
}

function indent(input, outdent) {
  const {value, selectionStart: start, selectionEnd: end} = input;
  const [first, last] = selectedLines(input);
  const rows = parse(value);
  const top = value.slice(0, first).split("\n").length - 1;
  const selected = rows.slice(top, top + value.slice(first, last).split("\n").length);
  if (!outdent && start === end && !selected[0].match && !LIST_ITEM.test(selected[0].line)) {
    edit(input, start, end, INDENT, start + INDENT.length);
    return;
  }
  // Each list item moves a level, with the lines under it. Tab moves an item and the deeper
  // lines after it together, in under its previous sibling; Shift+Tab moves each item out, along
  // with its parent when that moves too, else to its parent's indent. Lines before the first item
  // move with it; with no item at all, every line moves by INDENT, as before.
  const moves = [];
  let move, anchor;
  selected.forEach(({match, margin}, index) => {
    if (match && outdent) {
      const {parent} = relatives(rows, top + index, margin);
      const along = parent && parent.row >= top ? moves[parent.row - top] : 0;
      move = along || outer(parent, margin) - margin;
    } else if (match && (anchor === undefined || margin < anchor)) {
      anchor = margin;
      move = level(rows, top + index, margin) - margin;
    }
    moves.push(move);
  });
  // Tab moves every selected item or, if one has nothing to nest under, none of them.
  if (!outdent && moves.some((shift, index) => shift === 0 && selected[index].match)) moves.fill(0);
  const lead = moves.find((shift) => shift !== undefined) ?? (outdent ? -TAB : TAB);
  const tabs = tabbed(rows);
  const lines = selected.map(({line, space, margin}, index) => {
    const shift = moves[index] ?? lead;
    if (!shift || (selected.length > 1 && !line.trim())) return line;
    return reach(space, Math.max(0, margin + shift), tabs) + line.slice(space.length);
  });
  const changed = lines.join("\n");
  if (changed === value.slice(first, last)) return;
  // A position in a line's indent stays within the new indent; past it, it moves with the text.
  const map = (position) => {
    let from = first, to = first;
    for (let index = 0; index < selected.length; index += 1) {
      const {line, space} = selected[index];
      const indented = lines[index].length - (line.length - space.length);
      if (position <= from + line.length) {
        const column = position - from;
        return to + (column < space.length ? Math.min(column, indented) : column - space.length + indented);
      }
      from += line.length + 1;
      to += lines[index].length + 1;
    }
    return position + to - from;
  };
  edit(input, first, last, changed, map(start), map(end));
}

function continueList(input) {
  const {value, selectionStart: start, selectionEnd: end} = input;
  if (start !== end) return false;
  const first = start && value.lastIndexOf("\n", start - 1) + 1;
  const next = value.indexOf("\n", start);
  const last = next === -1 ? value.length : next;
  const match = value.slice(first, last).match(LIST_ITEM);
  if (!match || start < first + match[0].length) return false;
  const [marker, space, bullet, number, delimiter, gap, task] = match;
  if (!value.slice(first + marker.length, last).trim()) {
    // Enter on an empty item: move a nested item out a level, otherwise end the list.
    const margin = columns(space);
    if (margin) {
      const rows = parse(value);
      const row = value.slice(0, first).split("\n").length - 1;
      const out = reach(space, outer(relatives(rows, row, margin).parent, margin), tabbed(rows));
      edit(input, first, first + space.length, out, last - space.length + out.length);
    } else {
      edit(input, first, last, "", first);
    }
    return true;
  }
  const insertion = `\n${space}${number === undefined ? bullet : `${Number(number) + 1}${delimiter}`}${gap}${task ? "[ ] " : ""}`;
  edit(input, start, end, insertion, start + insertion.length);
  return true;
}

function toggle(input, mark) {
  const {value, selectionStart: start, selectionEnd: end} = input;
  const width = mark.length;
  const selected = value.slice(start, end);
  // A single * beside ** belongs to the bold marks, not to italics.
  const lone = (at) => mark !== "*" || value[at] !== "*";
  if (value.slice(start - width, start) === mark && value.slice(end, end + width) === mark
      && lone(start - width - 1) && lone(end + width)) {
    edit(input, start - width, end + width, selected, start - width, end - width);
  } else if (selected.length >= 2 * width && selected.startsWith(mark) && selected.endsWith(mark)) {
    const inner = selected.slice(width, -width);
    edit(input, start, end, inner, start, start + inner.length);
  } else {
    edit(input, start, end, `${mark}${selected}${mark}`, start + width, end + width);
  }
}

function link(input) {
  const {value, selectionStart: start, selectionEnd: end} = input;
  const selected = value.slice(start, end);
  if (/^https?:\/\/\S+$/.test(selected)) {
    edit(input, start, end, `[](${selected})`, start + 1);
  } else {
    edit(input, start, end, `[${selected}]()`, selected ? start + selected.length + 3 : start + 1);
  }
}

function markdownKeys(input) {
  input.addEventListener("keydown", (event) => {
    if (event.isComposing || event.altKey) return;
    const modified = event.metaKey || event.ctrlKey;
    const command = MAC ? event.metaKey : event.ctrlKey;
    const key = event.key.toLowerCase();
    const {selectionStart: start, selectionEnd: end, value} = input;
    let handled = true;
    if (event.key === "Tab" && !modified) indent(input, event.shiftKey);
    else if (event.key === "Enter" && !modified && !event.shiftKey) handled = continueList(input);
    else if (command && !event.shiftKey && key === "b") toggle(input, "**");
    else if (command && !event.shiftKey && key === "i") toggle(input, "*");
    else if (command && !event.shiftKey && key === "k") link(input);
    else if (!modified && PAIRS[event.key] && start !== end) {
      edit(input, start, end, `${event.key}${value.slice(start, end)}${PAIRS[event.key]}`, start + 1, end + 1);
    } else handled = false;
    if (handled) event.preventDefault();
  });
}

// The editor, replacing the contents of `host`. Given `title`, a title field comes first.
// save({text, title}) persists the changes; close(text) returns to the view.
export function showEditor(host, {key, text: original, title: originalTitle, empty, save, close}) {
  const titled = originalTitle !== undefined;
  const title = element("input", "editor-title");
  title.value = originalTitle ?? "";
  title.placeholder = "Title";
  title.setAttribute("aria-label", "Title");
  const input = element("textarea", "editor-input");
  input.value = original;
  input.placeholder = "Markdown, with math in $...$ and $$...$$; paste or drop images";
  const preview = element("div", "writeup-view editor-preview");
  const note = element("span", "editor-status");
  const saveButton = button("Save", "primary", () => commit());
  let frame = 0;
  let uploads = 0;

  const update = () => { preview.innerHTML = input.value.trim() ? render(input.value) : empty; };
  const grow = () => {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight + 2}px`;
  };
  const changed = () => {
    if ((titled && title.value.trim() !== originalTitle) || input.value !== original) dirty.add(key);
    else dirty.delete(key);
    note.textContent = dirty.has(key) ? "Unsaved changes" : "";
    grow();
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(update);
  };
  const cancel = () => {
    if (dirty.has(key) && !confirm("Discard unsaved changes?")) return;
    dirty.delete(key);
    close(original);
  };
  async function commit() {
    const name = title.value.replace(/\s+/g, " ").trim();
    if (titled && !name) {
      note.textContent = "The title can't be empty";
      title.focus();
      return;
    }
    if (uploads) {
      note.textContent = "Wait for the images to finish uploading";
      return;
    }
    const text = input.value === original ? original : input.value.trim() ? `${input.value.trimEnd()}\n` : "";
    saveButton.disabled = true;
    note.textContent = "Saving…";
    try {
      await save({text, title: titled ? name : undefined});
      dirty.delete(key);
      close(text);
    } catch (error) {
      note.textContent = `Not saved: ${error.message}`;
      saveButton.disabled = false;
    }
  }
  async function insertImages(files) {
    for (const file of files) {
      const marker = `![Uploading ${++uploads > 1 ? uploads : ""}…]()`;
      const start = input.selectionStart;
      const before = start > 0 && input.value[start - 1] !== "\n" ? "\n" : "";
      input.setRangeText(`${before}${marker}\n`, start, input.selectionEnd, "end");
      changed();
      let replacement = "";
      try {
        replacement = `![](${await uploadImage(file)})`;
      } catch (error) {
        note.textContent = `Image not uploaded: ${error.message}`;
      }
      const at = input.value.indexOf(marker);
      if (at >= 0) input.setRangeText(replacement, at, at + marker.length, "preserve");
      uploads -= 1;
      changed();
    }
  }
  const images = (list) => [...list].filter((file) => file.type.startsWith("image/"));

  title.addEventListener("input", changed);
  input.addEventListener("input", changed);
  markdownKeys(input);
  input.addEventListener("paste", (event) => {
    const files = images(event.clipboardData?.files ?? []);
    if (!files.length) return;
    event.preventDefault();
    insertImages(files);
  });
  input.addEventListener("dragover", (event) => {
    if ([...event.dataTransfer.items].some((item) => item.kind === "file")) event.preventDefault();
  });
  input.addEventListener("drop", (event) => {
    const files = images(event.dataTransfer.files);
    if (!files.length) return;
    event.preventDefault();
    input.focus();
    insertImages(files);
  });

  const panes = element("div", "editor-panes");
  panes.append(input, preview);
  const actions = element("div", "editor-actions");
  actions.append(note, button("Cancel", "ghost", cancel), saveButton);
  const editor = element("div", "editor");
  if (titled) editor.append(title);
  editor.append(panes, actions);
  editor.addEventListener("keydown", (event) => {
    if (((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s")
        || (event.target === title && event.key === "Enter")) {
      event.preventDefault();
      commit();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });
  host.replaceChildren(editor);
  update();
  grow();
  input.focus();
}
