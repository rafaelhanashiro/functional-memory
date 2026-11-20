// Render Markdown and math in the public results snapshot.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;
const OPEN = String.fromCharCode(0xe000);  // private-use characters mark math in the Markdown
const CLOSE = String.fromCharCode(0xe001);
const PLACEHOLDER = new RegExp(`${OPEN}(\\d+)${CLOSE}`, "g");
const MARKERS = /^(?:[ \t]*(?:>|(?:[-*+]|\d{1,9}[.)])(?=[ \t])))*/;  // quote and list markers opening a line
const QUOTE = /[ \t]*>[ \t]?/y;

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
