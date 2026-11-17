# Results site

A minimal static site for showing results, served on Kennel's loopback interface by
`serve.py`, which answers only its own Unix user:

```bash
nohup setsid python3 site/serve.py --port 8765 > <log> 2>&1 < /dev/null &
```

View it through `ssh -fN -o ExitOnForwardFailure=yes -L 8765:127.0.0.1:8765 rafah@kennel.csail.mit.edu`
at http://127.0.0.1:8765.

Literature write-ups are the user's own Markdown files in `literature/`, one per paper
(math in `$...$` and `$$...$$`), rendered when the paper's item opens. The user edits them
on the page; saving writes the file through `serve.py`, without committing it.

Its content changes only on the user's explicit instruction: do not add, update, or
refresh anything here otherwise, even when newer results exist.
