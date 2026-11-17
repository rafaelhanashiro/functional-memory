"""Serve the results site on Kennel's loopback interface and save edited write-ups.

The login node is shared and every user can reach 127.0.0.1, so only connections from
this server's own Unix user are answered. PUT /literature/<name>.md replaces an existing
write-up with the request body.

    nohup setsid python3 site/serve.py --port 8765 > <log> 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
from pathlib import Path
import re
import tempfile
import urllib.parse

SITE = Path(__file__).resolve().parent
WRITEUP = re.compile(r"/literature/([A-Za-z0-9._-]+\.md)")
MAX_BYTES = 1 << 20
LOOPBACK = "0100007F"


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


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _same_user(self) -> bool:
        if peer_uid(self.client_address[1], self.server.server_address[1]) == os.getuid():
            return True
        self.send_error(403)
        return False

    def do_GET(self):
        if self._same_user():
            super().do_GET()

    def do_HEAD(self):
        if self._same_user():
            super().do_HEAD()

    def do_PUT(self):
        if not self._same_user():
            return
        match = WRITEUP.fullmatch(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path))
        target = SITE / "literature" / match.group(1) if match else None
        if target is None or not target.is_file():
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BYTES:
            self.send_error(413)
            return
        try:
            text = self.rfile.read(length).decode("utf-8")
        except UnicodeDecodeError:
            self.send_error(400, "Write-ups must be UTF-8")
            return
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, prefix=".",
                                         suffix=".tmp", delete=False) as handle:
            handle.write(text)
        os.chmod(handle.name, 0o644)
        os.replace(handle.name, target)
        self.send_response(204)
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    handler = functools.partial(Handler, directory=str(SITE))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
