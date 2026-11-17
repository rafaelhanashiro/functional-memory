// Each paper's write-up is a Markdown file (math in $...$, $$...$$, \(...\), \[...\]),
// fetched every time its item opens. "Edit" shows the Markdown beside a live preview;
// "Save" sends it to serve.py, which writes the file. Code spans and blocks are matched
// first so their dollar signs stay literal.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;
const EMPTY = '<p class="empty">No write-up yet.</p>';
const papers = [...document.querySelectorAll(".paper[data-writeup]")];
const dirty = new Set();

function render(markdown) {
  const math = [];
  const source = markdown.replace(MATH, (match, code, display, bracket, paren, inline) => {
    if (code !== undefined) return code;
    const block = display !== undefined || bracket !== undefined;
    math.push(katex.renderToString(display ?? bracket ?? paren ?? inline, {displayMode: block, throwOnError: false}));
    return `${math.length - 1}`;
  });
  return marked.parse(source).replace(/(\d+)/g, (match, index) => math[index]);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text) node.textContent = text;
  return node;
}

function button(label, kind, onClick) {
  const node = element("button", `button ${kind}`, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function showView(paper, text) {
  dirty.delete(paper);
  const view = element("div", "writeup-view");
  view.innerHTML = text.trim() ? render(text) : EMPTY;
  paper.querySelector(".writeup").replaceChildren(button("Edit", "edit", () => showEditor(paper, text)), view);
}

function showEditor(paper, original) {
  const input = element("textarea", "editor-input");
  input.value = original;
  input.placeholder = "Markdown, with math in $...$ and $$...$$";
  const preview = element("div", "writeup-view editor-preview");
  const status = element("span", "editor-status");
  const saveButton = button("Save", "primary", () => save());
  let frame = 0;

  const update = () => { preview.innerHTML = input.value.trim() ? render(input.value) : EMPTY; };
  const grow = () => {
    input.style.height = "auto";
    input.style.height = `${input.scrollHeight + 2}px`;
  };
  const cancel = () => {
    if (!dirty.has(paper) || confirm("Discard unsaved changes?")) showView(paper, original);
  };
  async function save() {
    const text = input.value.trim() ? `${input.value.trimEnd()}\n` : "";
    saveButton.disabled = true;
    status.textContent = "Saving…";
    try {
      const response = await fetch(paper.dataset.writeup, {
        method: "PUT", body: text, headers: {"Content-Type": "text/markdown; charset=utf-8"},
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      showView(paper, text);
    } catch (error) {
      status.textContent = `Not saved: ${error.message}`;
      saveButton.disabled = false;
    }
  }

  input.addEventListener("input", () => {
    if (input.value === original) dirty.delete(paper);
    else dirty.add(paper);
    status.textContent = dirty.has(paper) ? "Unsaved changes" : "";
    grow();
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(update);
  });
  input.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
      event.preventDefault();
      save();
    } else if (event.key === "Escape") {
      event.preventDefault();
      cancel();
    }
  });

  const panes = element("div", "editor-panes");
  panes.append(input, preview);
  const actions = element("div", "editor-actions");
  actions.append(status, button("Cancel", "ghost", cancel), saveButton);
  const editor = element("div", "editor");
  editor.append(panes, actions);
  paper.querySelector(".writeup").replaceChildren(editor);
  update();
  grow();
  input.focus();
}

for (const paper of papers) {
  paper.addEventListener("toggle", async () => {
    if (!paper.open) return;
    try {
      const response = await fetch(paper.dataset.writeup, {cache: "no-store"});
      if (!response.ok) throw new Error(response.statusText);
      showView(paper, await response.text());
    } catch {
      paper.querySelector(".writeup").innerHTML = '<p class="empty">Could not load the write-up.</p>';
    }
  });
  paper.querySelector("summary").addEventListener("click", (event) => {
    if (paper.open && dirty.has(paper)) {
      if (confirm("Discard unsaved changes?")) dirty.delete(paper);
      else event.preventDefault();
    }
  });
}

window.addEventListener("beforeunload", (event) => {
  if (dirty.size) event.preventDefault();
});
