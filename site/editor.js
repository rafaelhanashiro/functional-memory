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
