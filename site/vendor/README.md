# Vendored libraries

Copied from jsDelivr so the site needs nothing outside this folder. Both are MIT licensed
(`LICENSE` files beside them).

| Library | Version | Files |
| --- | --- | --- |
| [marked](https://github.com/markedjs/marked) | 12.0.2 | `marked/marked.min.js` |
| [KaTeX](https://github.com/KaTeX/KaTeX) | 0.16.11 | `katex/katex.min.js`, `katex/katex.min.css`, `katex/fonts/*.woff2` |

`katex.min.css` keeps only the WOFF2 font sources: each `@font-face` drops its WOFF and
TrueType fallbacks, which every current browser skips, and only the WOFF2 fonts are copied.
