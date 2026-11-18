"""Serve the results site on Kennel's loopback interface and save literature edits.

The login node is shared and every user can reach 127.0.0.1, so only connections from
this server's own Unix user are answered. Besides static files it accepts:

    PUT   /literature/<id>.md      replace an existing write-up with the request body
    PUT   /meeting/agenda.md        replace the meeting agenda (likewise notes.md)
    POST  /api/media               an image body (PNG, JPEG, GIF, WebP): store it in media/
    PUT   /api/literature          {"order": [ids]}: reorder the resources (the page's order)
    POST  /api/literature          {"url": ...}: add a resource, reading its title, authors and year
    PATCH /api/literature/<id>     {"seen": true or false} moves a resource between the two
                                   groups; {"title": ...} renames it

    nohup setsid python3 site/serve.py --port 8765 > <log> 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import functools
from html.parser import HTMLParser
import http.server
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
import threading
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SITE = Path(__file__).resolve().parent
LITERATURE = SITE / "literature"
LIST = LITERATURE / "papers.json"
DOCUMENT = re.compile(r"/(literature/[A-Za-z0-9._-]+\.md|meeting/(?:agenda|notes)\.md)")
RESOURCE = re.compile(r"/api/literature/([A-Za-z0-9._-]+)")
ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}|[a-z-]+(?:\.[a-z]{2})?/[0-9]{7})", re.I)
ATOM = {"a": "http://www.w3.org/2005/Atom"}
AGENT = "Mozilla/5.0 (functional-memory literature list)"
MEDIA = SITE / "media"
IMAGES = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
MAX_BYTES = 1 << 20
MAX_IMAGE_BYTES = 25 << 20
FETCH_BYTES = 4 << 20
LOOPBACK = "0100007F"
LOCK = threading.Lock()


def peer_uid(client_port: int, server_port: int) -> int | None:
    """The Unix user owning the loopback TCP socket that connected to this server."""
    with open("/proc/net/tcp") as table:
        next(table)
        for line in table:
            fields = line.split()
            if (fields[1] == f"{LOOPBACK}:{client_port:04X}"
                    and fields[2] == f"{LOOPBACK}:{server_port:04X}"):
                return int(fields[7])
    return None


def write_atomic(path: Path, content: str | bytes) -> None:
    data = content.encode("utf-8") if isinstance(content, str) else content
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=".", suffix=".tmp", delete=False) as handle:
        handle.write(data)
    os.chmod(handle.name, 0o644)
    os.replace(handle.name, path)


def load_list() -> list[dict]:
    return json.loads(LIST.read_text()) if LIST.exists() else []


def save_list(papers: list[dict]) -> None:
    write_atomic(LIST, json.dumps(papers, indent=2, ensure_ascii=False) + "\n")


def fetch(url: str) -> tuple[bytes, str, str]:
    request = urllib.request.Request(url, headers={"User-Agent": AGENT})
    with urllib.request.urlopen(request, timeout=20) as response:
        return (response.read(FETCH_BYTES), response.headers.get_content_type(),
                response.headers.get_content_charset() or "utf-8")


def arxiv_metadata(identifier: str) -> dict:
    data, _, _ = fetch(f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(identifier)}")
    entry = ET.fromstring(data).find("a:entry", ATOM)
    if entry is None or "api/errors" in entry.findtext("a:id", "", ATOM):
        raise ValueError(f"arXiv has no paper {identifier}")
    return {
        "title": " ".join(entry.findtext("a:title", "", ATOM).split()),
        "authors": [" ".join(author.findtext("a:name", "", ATOM).split())
                    for author in entry.findall("a:author", ATOM)],
        "year": int(entry.findtext("a:published", "", ATOM)[:4]),
        "source": "arXiv",
    }


class PageMetadata(HTMLParser):
    """Collects <meta> tags and the <title> of an HTML page."""

    def __init__(self):
        super().__init__()
        self.meta: dict[str, list[str]] = {}
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        key = (attrs.get("name") or attrs.get("property") or "").lower()
        if tag == "meta" and key and attrs.get("content"):
            self.meta.setdefault(key, []).append(" ".join(attrs["content"].split()))
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


def author_name(name: str) -> str:
    """'Roy, Aurko' (citation metadata) becomes 'Aurko Roy'."""
    last, comma, first = name.partition(",")
    return f"{first.strip()} {last.strip()}" if comma and first.strip() and "," not in first else name


def link_metadata(url: str) -> dict:
    """Title, authors, year and site from a page's citation, Open Graph or <title> tags."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or url).removeprefix("www.")
    fallback = urllib.parse.unquote(parts.path.rstrip("/").rsplit("/", 1)[-1]) or host
    metadata = {"title": fallback, "authors": [], "year": None, "source": host}
    data, kind, charset = fetch(url)
    if kind not in ("text/html", "application/xhtml+xml"):
        return metadata
    page = PageMetadata()
    page.feed(data.decode(charset, "replace"))
    first = lambda *keys: next((page.meta[key][0] for key in keys if page.meta.get(key)), "")
    title = first("citation_title", "og:title", "twitter:title", "dc.title") or " ".join(page.title.split())
    published = first("citation_publication_date", "citation_date", "citation_online_date",
                      "article:published_time", "dc.date")
    year = re.search(r"(?:19|20)\d\d", published)
    authors = page.meta.get("citation_author") or page.meta.get("dc.creator") or []
    return {
        "title": title or fallback,
        "authors": [author_name(author) for author in authors],
        "year": int(year.group()) if year else None,
        "source": first("og:site_name") or host,
    }


def same_link(first: str, second: str) -> bool:
    normalize = lambda url: urllib.parse.urlsplit(url)._replace(scheme="", fragment="").geturl().rstrip("/")
    return normalize(first) == normalize(second)


def new_id(base: str, papers: list[dict]) -> str:
    base = re.sub(r"[^a-z0-9.]+", "-", base.lower()).strip("-.")
    if len(base) > 60:
        base = base[:61].rsplit("-", 1)[0]
    base = base.strip("-.") or "resource"
    taken = {paper["id"] for paper in papers}
    candidate, number = base, 2
    while candidate in taken or (LITERATURE / f"{candidate}.md").exists():
        candidate, number = f"{base}-{number}", number + 1
    return candidate


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _same_user(self) -> bool:
        if peer_uid(self.client_address[1], self.server.server_address[1]) == os.getuid():
            return True
        self.send_error(403)
        return False

    def _route(self) -> str:
        return urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)

    def _body(self) -> bytes | None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BYTES:
            self.send_error(413)
            return None
        return self.rfile.read(length)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self._same_user():
            super().do_GET()

    def do_HEAD(self):
        if self._same_user():
            super().do_HEAD()

    def do_PUT(self):
        if not self._same_user():
            return
        if self._route() == "/api/literature":
            self._reorder()
            return
        match = DOCUMENT.fullmatch(self._route())
        target = SITE / match.group(1) if match else None
        if target is None or not target.is_file():
            self.send_error(404)
            return
        body = self._body()
        if body is None:
            return
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError:
            self.send_error(400, "Documents must be UTF-8")
            return
        write_atomic(target, text)
        self.send_response(204)
        self.end_headers()

    def _reorder(self):
        body = self._body()
        if body is None:
            return
        try:
            order = json.loads(body)["order"]
        except (ValueError, KeyError, TypeError):
            order = None
        if not isinstance(order, list) or not all(isinstance(identifier, str) for identifier in order):
            self._json(400, {"error": 'Expected {"order": [ids]}'})
            return
        with LOCK:
            papers = {paper["id"]: paper for paper in load_list()}
            if len(order) != len(papers) or set(order) != set(papers):
                self._json(409, {"error": "The list changed since the page loaded; reload it"})
                return
            save_list([papers[identifier] for identifier in order])
        self._json(200, {"order": order})

    def _upload(self):
        kind = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        length = int(self.headers.get("Content-Length") or 0)
        if kind not in IMAGES:
            self._json(415, {"error": "Only PNG, JPEG, GIF and WebP images can be added"})
            return
        if not 0 < length <= MAX_IMAGE_BYTES:
            self._json(413, {"error": f"Images must be at most {MAX_IMAGE_BYTES >> 20} MB"})
            return
        data = self.rfile.read(length)
        MEDIA.mkdir(exist_ok=True)
        name = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}{IMAGES[kind]}"
        write_atomic(MEDIA / name, data)
        self._json(201, {"path": f"media/{name}"})

    def do_POST(self):
        if not self._same_user():
            return
        if self._route() == "/api/media":
            self._upload()
            return
        if self._route() != "/api/literature":
            self.send_error(404)
            return
        body = self._body()
        if body is None:
            return
        try:
            url = json.loads(body)["url"].strip()
        except (ValueError, KeyError, TypeError, AttributeError):
            self._json(400, {"error": 'Expected {"url": ...}'})
            return
        if urllib.parse.urlsplit(url).scheme not in ("http", "https") or not urllib.parse.urlsplit(url).hostname:
            self._json(400, {"error": "The link must be an http or https address"})
            return
        arxiv = ARXIV.search(url)
        identifier = arxiv.group(1) if arxiv else None
        try:
            try:
                metadata = arxiv_metadata(identifier) if identifier else link_metadata(url)
            except Exception:
                if not identifier:
                    raise
                metadata = link_metadata(f"https://arxiv.org/abs/{identifier}") | {"source": "arXiv"}
        except Exception as error:
            self._json(502, {"error": f"Could not read the link: {error}"})
            return
        with LOCK:
            papers = load_list()
            for paper in papers:
                if (identifier and paper.get("arxiv") == identifier) or same_link(paper["url"], url):
                    self._json(409, {"error": f"Already listed: {paper['title']}"})
                    return
            entry = {"id": new_id(identifier or metadata["title"], papers), **metadata, "url": url,
                     **({"arxiv": identifier} if identifier else {}), "seen": False,
                     "added": date.today().isoformat()}
            (LITERATURE / f"{entry['id']}.md").touch()
            save_list([entry, *papers])
        self._json(201, entry)

    def do_PATCH(self):
        if not self._same_user():
            return
        match = RESOURCE.fullmatch(self._route())
        if not match:
            self.send_error(404)
            return
        body = self._body()
        if body is None:
            return
        try:
            changes = json.loads(body)
        except ValueError:
            changes = None
        if not isinstance(changes, dict) or not changes or set(changes) - {"seen", "title"}:
            self._json(400, {"error": 'Expected {"seen": true or false} or {"title": ...}'})
            return
        if "seen" in changes and not isinstance(changes["seen"], bool):
            self._json(400, {"error": "seen must be true or false"})
            return
        if "title" in changes:
            title = " ".join(changes["title"].split()) if isinstance(changes["title"], str) else ""
            if not title:
                self._json(400, {"error": "The title can't be empty"})
                return
            changes["title"] = title
        with LOCK:
            papers = load_list()
            entry = next((paper for paper in papers if paper["id"] == match.group(1)), None)
            if entry is None:
                self._json(404, {"error": f"No resource {match.group(1)}"})
                return
            entry.update(changes)
            save_list(papers)
        self._json(200, entry)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(SITE))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
