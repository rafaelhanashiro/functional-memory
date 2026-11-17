# Results site

A minimal static site for showing results, served on Kennel's loopback interface by
`serve.py`, which answers only its own Unix user:

```bash
nohup setsid python3 site/serve.py --port 8765 > <log> 2>&1 < /dev/null &
```

View it through `ssh -fN -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 rafah@kennel.csail.mit.edu`
at http://127.0.0.1:8765.

`literature/papers.json` lists the literature resources (title, authors, year, link, and
whether the user has seen it). The page adds a resource from a link, reading its metadata,
and switches it between seen and not seen; each resource's write-up is the user's own
Markdown file `literature/<id>.md` (math in `$...$` and `$$...$$`), edited on the page.
All of these go through `serve.py`, which writes the files without committing them.

Its content changes only on the user's explicit instruction: do not add, update, or
refresh anything here otherwise, even when newer results exist.
