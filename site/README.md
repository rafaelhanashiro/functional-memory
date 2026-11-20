# Results site

A minimal static site for showing results, served on Kennel's loopback interface by
`serve.py`, which answers only its own Unix user:

```bash
nohup setsid python3 site/serve.py --port 8765 > <log> 2>&1 < /dev/null &
```

View it through `ssh -fN -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 rafah@kennel.csail.mit.edu`
at http://127.0.0.1:8765. Elsewhere, such as a laptop with Python 3.9 or later, run
`python3 site/serve.py --port 8766` from the repository root and open http://127.0.0.1:8766
(another port while a tunnel holds 8765); edits there are saved in that checkout.

Everything the pages load is inside this folder: images in `media/`, result figures in `figures/`, result data in `data/`,
and the Markdown and math libraries in `vendor/`. Only adding a literature link reaches the
internet, to read the link's title.

`literature/papers.json` lists the literature resources in the page's order (title,
authors, year, link, and whether the user has seen it). On the page the user adds a
resource from a link (it goes to the top, with metadata read from the link), reorders,
renames, and switches resources between seen and not seen, and edits each resource's
write-up, their own Markdown file `literature/<id>.md` (math in `$...$` and `$$...$$`).
The Meetings page lists the meetings by date, newest first, and adds one for a chosen date,
optionally named. A meeting's page (`meeting.html?date=YYYY-MM-DD`) shows its agenda and
notes, `meeting/<date>-agenda.md` and `meeting/<date>-notes.md`, edited the same way; the
PDF button on either prints it alone, for the print dialog's Save as PDF. Its Edit meeting
button renames the meeting (the name is kept in `meeting/<date>.json`), moves it to another
date, or deletes it after asking. Images pasted or dropped into any editor are stored in `media/` and
linked from the Markdown; saving any document deletes the images that nothing links to
once they are more than a day old. Commit only images that are linked. All of these go through `serve.py`, which writes the files
without committing them.

The result pages draw their tables from `data/*.json`, each written by the generator beside
it from the Transformer++ runs' records and result summaries: `polynomial.py` (Polynomial, with
its chart), `other.py` (Other runs), `scaling.py` (2.5× Chinchilla, 4k sequences and 100M),
`parametrizations.py` (Activations, Dense, and SoS / Softplus), `alternatives.py` (Fixed
kernels), and `diagnostics.py` (Overfitting and Continued pretraining; it also copies the Length
extrapolation figures from their result collection into `figures/`). `results.js` draws the
tables of all but the first two from cells the generators format.

The result pages' text and titles are the user's, edited on the page: the pencil in the header
switches a page to editing, where clicking a text, or an empty place for one, opens the shared
editor, and clicking a title edits it in place (`text.js`). There is a place for text under each
page's title, over each table or figure (an intro), and under it (notes); each is Markdown with
math in `text/<page>/<name>.md`, which `serve.py` lists for the page and saves, deleting a text
the user empties. The titles are the page's own heading, its section headings (`data-title`),
and the banners naming other pages; a renamed one is kept in `text/<page>/titles.json` (key
`title` for the page itself, which also renames its banners, back links, and browser tab), and
clearing it restores the original. Like the meetings' files, these are the user's once written:
never overwrite, regenerate, or delete them, and commit their edits only when the user asks. New
content the user asks for may add new texts. Rerun a generator only when the user asks for newer
results. Every perplexity on the site is the run's validation-best value, never the terminal
one; the overfitting fits report perplexity on the batch they fit. The Learned polynomials
figure in `figures/` is a copy with no generator.

The meetings' files (`meeting/<date>-agenda.md`, `meeting/<date>-notes.md`,
`meeting/<date>.json`, and the images they link in `media/`) belong to the user, who edits
them live: never write, reset, regenerate, or stage them; commit them only when the user asks.

A run at another model size or token budget belongs on that scaling's page under Other scalings
(100M, 2.5× Chinchilla, 4k sequences), whatever its mechanism; the reference shape's 1× runs go on
their mechanism's page (user, 2026-11-19).

Its content changes only on the user's explicit instruction: do not add, update, or
refresh anything here otherwise, even when newer results exist.
