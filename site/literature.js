// Each paper's write-up is a Markdown file (math in $...$, $$...$$, \(...\), \[...\]),
// fetched and rendered every time its item opens, so edits show on the next open.
// Code spans and blocks are matched first and left untouched.
const MATH = /(```[\s\S]*?```|`[^`\n]+`)|\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]|\\\(([\s\S]+?)\\\)|(?<![\\$])\$(?!\s)((?:\\.|[^$\\\n])+?)(?<!\s)\$/g;

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

for (const paper of document.querySelectorAll(".paper[data-writeup]")) {
  paper.addEventListener("toggle", async () => {
    if (!paper.open) return;
    const body = paper.querySelector(".writeup");
    try {
      const response = await fetch(paper.dataset.writeup, {cache: "no-store"});
      if (!response.ok) throw new Error(response.statusText);
      const text = (await response.text()).trim();
      body.innerHTML = text ? render(text) : '<p class="empty">No write-up yet.</p>';
    } catch {
      body.innerHTML = '<p class="empty">Could not load the write-up.</p>';
    }
  });
}
