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

Everything the pages load is inside this folder: images in `media/`, result data in `data/`,
and the Markdown and math libraries in `vendor/`. Only adding a literature link reaches the
internet, to read the link's title.

`literature/papers.json` lists the literature resources in the page's order (title,
authors, year, link, and whether the user has seen it). On the page the user adds a
resource from a link (it goes to the top, with metadata read from the link), reorders,
renames, and switches resources between seen and not seen, and edits each resource's
write-up, their own Markdown file `literature/<id>.md` (math in `$...$` and `$$...$$`).
The Meetings page lists the meetings by date, newest first, and adds one for a chosen date.
A meeting's page (`meeting.html?date=YYYY-MM-DD`) shows its agenda and notes,
`meeting/<date>-agenda.md` and `meeting/<date>-notes.md`, edited the same way; the PDF
button on either prints it alone, for the print dialog's Save as PDF. Images pasted or dropped into any editor are stored in `media/` and
linked from the Markdown; saving any document deletes the images that nothing links to
once they are more than a day old. Commit only images that are linked. All of these go through `serve.py`, which writes the files
without committing them.

The Polynomial runs page draws its tables and chart from `data/polynomial.json`, which
`data/polynomial.py` writes from the Transformer++ result summaries; rerun it only when the
user asks for newer results. Every perplexity on the site is the run's validation-best value,
never the terminal one.

The meetings' files (`meeting/<date>-agenda.md`, `meeting/<date>-notes.md`, and the images
they link in `media/`) belong to the user, who edits them live: never write, reset,
regenerate, or stage them; commit them only when the user asks.

Its content changes only on the user's explicit instruction: do not add, update, or
refresh anything here otherwise, even when newer results exist.
