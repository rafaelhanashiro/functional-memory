#!/usr/bin/env python3
"""Publish the read-only results snapshot that GitHub Pages serves from the gh-pages branch.

The snapshot is the committed site/ without its private and editing parts: no Literature, Meetings,
Plan or Notes pages (nor links to them), no serve.py, media, page-text sources, data generators or READMEs.
Page texts and renamed titles are baked into data/page-text.json. public/ holds the modules that
replace the editing ones: a read-only text.js, and markdown.js, which re-exports editor.js's
rendering.

    python3 site/publish.py               build from HEAD and push it to gh-pages
    python3 site/publish.py --build DIR   only build the snapshot into DIR, to look at it

Only committed files are published. The new gh-pages commit goes on top of origin/gh-pages (no
force push), under this checkout's git identity; GitHub Pages rebuilds the site a minute later.
"""
import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SITE_URL = "https://rafaelhanashiro.github.io/functional-memory/"
ROOT = Path(__file__).resolve().parent.parent
PRIVATE_FILES = {"README.md", "serve.py", "publish.py", "text.js", "literature.html", "literature.js", "meeting.html",
                 "meeting.js", "queue.html", "queue.js", "notes.html", "vendor/README.md"}
PRIVATE_DIRS = {"literature", "meeting", "queue", "text", "media", "public", "__pycache__"}
NAV = ['      <a href="literature.html">Literature</a>\n', '      <a href="meeting.html">Meetings</a>\n',
       '      <a href="queue.html">Plan</a>\n', '      <a href="notes.html">Notes</a>\n']


def git(*args, stdin=None, env=None):
    result = subprocess.run(["git", *args], cwd=ROOT, input=stdin, env=env, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def build(site, out):
    """Write the snapshot of the site/ folder ``site`` into the empty folder ``out``."""
    for source in sorted(site.rglob("*")):
        rel = source.relative_to(site).as_posix()
        if source.is_dir() or rel in PRIVATE_FILES or rel.split("/")[0] in PRIVATE_DIRS:
            continue
        if rel.startswith("data/") and rel.endswith(".py") or rel.endswith(".pyc"):
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not rel.endswith(".html"):
            shutil.copyfile(source, target)
            continue
        html = source.read_text(encoding="utf-8")
        for line in NAV:
            if html.count(line) != 1:
                sys.exit(f"{rel}: expected one nav link {line.strip()!r}")
            html = html.replace(line, "")
        target.write_text(html, encoding="utf-8")
    for module in sorted((site / "public").glob("*.js")):
        shutil.copyfile(module, out / module.name)
    (out / ".nojekyll").write_text("")

    # Pages in the site's order: home, each section's pages and then its own index, then any other
    # published page that has texts.
    pages = ["index"]
    for section in ("runs", "diagnostics"):
        pages += sorted(path.stem for path in site.glob(f"{section}-*.html")) + [section]
    pages += sorted(path.stem for path in out.glob("*.html")
                    if path.stem not in pages and (site / "text" / path.stem).is_dir())
    texts, titles = {}, {}
    for page in pages:
        folder = site / "text" / page
        texts[page] = {path.stem: path.read_text(encoding="utf-8") for path in sorted(folder.glob("*.md"))}
        if (folder / "titles.json").exists():
            titles[page] = json.loads((folder / "titles.json").read_text(encoding="utf-8"))
    (out / "data" / "page-text.json").write_text(
        json.dumps({"pages": texts, "titles": titles}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def publish(out, scratch, trailer):
    """Commit the snapshot in ``out`` on top of origin/gh-pages and push it."""
    git("fetch", "-q", "origin", "+gh-pages:refs/remotes/origin/gh-pages")
    parent = git("rev-parse", "refs/remotes/origin/gh-pages")
    env = {**os.environ, "GIT_INDEX_FILE": str(scratch / "index")}
    entries = "".join(f"100644 {git('hash-object', '-w', str(path))}\t{path.relative_to(out).as_posix()}\n"
                      for path in sorted(out.rglob("*")) if path.is_file())
    git("update-index", "--add", "--index-info", stdin=entries, env=env)
    tree = git("write-tree", env=env)
    if tree == git("rev-parse", f"{parent}^{{tree}}"):
        print(f"gh-pages already shows this snapshot: {SITE_URL}")
        return
    message = f"Publish the read-only results snapshot of main {git('rev-parse', '--short', 'HEAD')}\n"
    if trailer:
        message += f"\n{trailer}\n"
    commit = git("commit-tree", tree, "-p", parent, stdin=message)
    git("push", "-q", "origin", f"{commit}:refs/heads/gh-pages")
    print(f"Pushed {commit[:8]} to gh-pages; GitHub Pages shows it in about a minute: {SITE_URL}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", metavar="DIR", type=Path, help="only build the snapshot into DIR")
    parser.add_argument("--trailer", help="a last paragraph for the gh-pages commit message")
    args = parser.parse_args()
    if args.build and args.build.exists() and any(args.build.iterdir()):
        sys.exit(f"{args.build} is not empty")
    if git("status", "--porcelain", "--", "site"):
        print("Note: site/ has uncommitted changes; only what HEAD holds is published.")
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        archive = subprocess.run(["git", "archive", "--format=tar", "HEAD:site"], cwd=ROOT,
                                 capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(scratch / "site")  # our own repository's files
        out = args.build or scratch / "out"
        out.mkdir(parents=True, exist_ok=True)
        build(scratch / "site", out)
        if args.build:
            print(f"Built the snapshot in {out}")
            return
        publish(out, scratch, args.trailer)


if __name__ == "__main__":
    main()
