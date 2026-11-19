// Shared by the pages: Markdown rendering with math ($...$, $$...$$, \(...\), \[...\]; code
// spans and blocks are matched first so their dollar signs stay literal), and the editor:
// Markdown beside a live preview, with images pasted or dropped into the text uploaded
// through serve.py.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;
const OPEN = String.fromCharCode(0xe000);  // private-use characters mark math in the Markdown
const CLOSE = String.fromCharCode(0xe001);
const PLACEHOLDER = new RegExp(`${OPEN}(\\d+)${CLOSE}`, "g");

// Sheets with unsaved edits: leaving the page asks first.
export const dirty = new Set();
window.addEventListener("beforeunload", (event) => {
  if (dirty.size) event.preventDefault();
});

export function render(markdown) {
  const math = [];
  const source = markdown.replace(MATH, (match, code, display, bracket, paren, inline) => {
    if (code !== undefined) return code;
    const block = display !== undefined || bracket !== undefined;
    math.push(katex.renderToString(display ?? bracket ?? paren ?? inline, {displayMode: block, throwOnError: false}));
    return `${OPEN}${math.length - 1}${CLOSE}`;
  });
  return marked.parse(source).replace(PLACEHOLDER, (match, index) => math[index]);
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

// The rendered text with an "Edit" button.
export function showView(host, text, {empty, onEdit}) {
  const view = element("div", "writeup-view");
  view.innerHTML = text.trim() ? render(text) : empty;
  host.replaceChildren(button("Edit", "edit", onEdit), view);
}

// Markdown keys in the editor's text box, as in Obsidian: Tab and Shift+Tab move a list item,
// or the selected lines starting at one, a level in or out of its list, and indent or outdent
// other lines; Enter continues a list and ends it on an empty item; Cmd+B, I and K (Ctrl off a
// Mac) toggle bold, italics and a link; and typing a bracket, $ or ` with text selected wraps
// it. Edits go through the browser's undo history.
const LIST_ITEM = /^( *)([-*+]|(\d+)([.)]))( +)(\[[ xX]\] +)?/;
const INDENT = "    ";  // four spaces nest a list item under any marker (-, 1., 10.)
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
  const first = value.lastIndexOf("\n", start - 1) + 1;
  const last = value.indexOf("\n", end);
  return [first, last === -1 ? value.length : last];
}

// A level of nesting, as the text's lists already nest: the smallest step from a list item to
// its parent (two spaces in lists typed that way), or four (INDENT) while nothing is nested.
// Fenced code doesn't count, nor an item indented so far that Markdown reads it as text.
function nesting(lines) {
  let step;
  let fenced = false;
  const open = [];  // [indent, content column] of the items a later line can nest under
  for (const line of lines) {
    if (/^ *(```|~~~)/.test(line)) fenced = !fenced;
    const match = !fenced && line.match(LIST_ITEM);
    if (!match) {
      if (/^\S/.test(line)) open.length = 0;  // unindented text ends the list
      continue;
    }
    const margin = match[1].length;
    while (open.length && margin < open.at(-1)[1]) open.pop();
    if (margin >= (open.at(-1)?.[1] ?? 0) + INDENT.length) continue;
    if (open.length) step = Math.min(step ?? Infinity, margin - open.at(-1)[0]);
    open.push([margin, margin + match[2].length + match[5].length]);
  }
  return step ?? INDENT.length;
}

// The list around the item indented by `depth` on line `row` of `lines`, from the lines above
// it: its parent (its row, indent and content column), its previous sibling's content column,
// and the indent of the items nested under that sibling.
function relatives(lines, row, depth) {
  const found = {};
  for (let above = row - 1; above >= 0; above -= 1) {
    const match = lines[above].match(LIST_ITEM);
    if (!match) {
      if (/^\S/.test(lines[above])) break;  // unindented text ends the list
      continue;
    }
    const margin = match[1].length;
    const content = margin + match[2].length + match[5].length;
    if (margin < depth) return {...found, parent: {row: above, margin, content}};
    if (margin === depth) {
      found.sibling ??= content;
      if (!depth) break;  // a top-level item has no parent to find
    } else if (found.sibling === undefined) found.children = Math.min(found.children ?? margin, margin);
  }
  return found;
}

// Where Tab moves the list item indented by `depth` on line `row`: in under its previous
// sibling (or under the item above, if not nested in it yet), level with the items already
// nested there, else a nesting step in, at a column where Markdown nests it. With nothing above
// to nest under, it stays put.
function level(lines, row, depth) {
  const {parent, sibling, children} = relatives(lines, row, depth);
  const under = sibling ?? (parent && depth < parent.content ? parent.content : undefined);
  if (under === undefined) return depth;
  // An item nests at the content column of the item above and up to three spaces past it.
  const target = children < under + INDENT.length ? children : depth + nesting(lines);
  return target >= under && target < under + INDENT.length ? target : under;
}

// Where Shift+Tab, or Enter on an empty item, moves a nested list item: out to its parent's indent.
function outer(lines, row, depth) {
  const {parent} = relatives(lines, row, depth);
  return parent ? parent.margin : Math.max(0, depth - INDENT.length);
}

function indent(input, outdent) {
  const {value, selectionStart: start, selectionEnd: end} = input;
  const [first, last] = selectedLines(input);
  const block = value.slice(first, last);
  const item = block.match(LIST_ITEM);  // the first line, if it's a list item
  if (!outdent && start === end && !item) {
    edit(input, start, end, INDENT, start + INDENT.length);
    return;
  }
  const all = value.split("\n");
  const row = value.slice(0, first).split("\n").length - 1;
  const lines = block.split("\n");
  const shifts = [];
  if (!outdent) {
    // The lines move with their first list item, a level in; lines outside a list by INDENT.
    const move = item ? level(all, row, item[1].length) - item[1].length : INDENT.length;
    for (const line of lines) shifts.push(lines.length > 1 && !line.trim() ? 0 : move);
  } else {
    // Each list item moves out a level: along with its parent when that moves too, else to its
    // parent's indent. Other lines move with the item above them, or out by INDENT.
    let move = -INDENT.length;
    lines.forEach((line, index) => {
      const match = line.match(LIST_ITEM);
      if (match) {
        const depth = match[1].length;
        const {parent} = relatives(all, row + index, depth);
        const along = parent && parent.row >= row ? shifts[parent.row - row] : 0;
        move = along || outer(all, row + index, depth) - depth;
      }
      shifts.push(-(line.startsWith("\t") ? 1 : Math.min(-move, line.match(/^ */)[0].length)));
    });
  }
  const changed = lines.map((line, index) => shifts[index] < 0 ? line.slice(-shifts[index])
    : " ".repeat(shifts[index]) + line).join("\n");
  if (changed === block) return;
  const total = shifts.reduce((sum, shift) => sum + shift, 0);
  const newStart = Math.max(first, start + shifts[0]);
  edit(input, first, last, changed, newStart, Math.max(newStart, end + (start === end ? shifts[0] : total)));
}

function continueList(input) {
  const {value, selectionStart: start, selectionEnd: end} = input;
  if (start !== end) return false;
  const first = value.lastIndexOf("\n", start - 1) + 1;
  const next = value.indexOf("\n", start);
  const last = next === -1 ? value.length : next;
  const match = value.slice(first, last).match(LIST_ITEM);
  if (!match || start < first + match[0].length) return false;
  const [marker, spaces, bullet, number, delimiter, gap, task] = match;
  if (!value.slice(first + marker.length, last).trim()) {
    // Enter on an empty item: move a nested item out a level, otherwise end the list.
    if (spaces) {
      const above = value.slice(0, first).split("\n");
      const removed = spaces.length - outer(above, above.length - 1, spaces.length);
      edit(input, first, first + removed, "", last - removed);
    } else {
      edit(input, first, last, "", first);
    }
    return true;
  }
  const insertion = `\n${spaces}${number === undefined ? bullet : `${Number(number) + 1}${delimiter}`}${gap}${task ? "[ ] " : ""}`;
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
