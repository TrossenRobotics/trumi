"""Serve the timecode sync page locally over HTTP.

:Usage:
    uv run python scripts/serve_timecode_sync.py
"""

import http.server
import logging
import socketserver
from functools import partial
from pathlib import Path

import click

DIRECTORY = Path(__file__).resolve().parent.parent / "assets" / "timecode_sync"

logger = logging.getLogger(__name__)


@click.command(help="Serve the timecode sync page locally over HTTP.")
@click.option(
    "--port",
    type=int,
    default=8000,
    help="Port to serve on (default: 8000).",
)
def main(port):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not (DIRECTORY / "index.html").is_file():
        raise click.ClickException(f"{DIRECTORY / 'index.html'} not found.")

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DIRECTORY))
    socketserver.TCPServer.allow_reuse_address = True
    try:
        httpd = socketserver.TCPServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise click.ClickException(f"{e}. Try a different --port.")
    with httpd:
        logger.info("Serving %s at http://127.0.0.1:%d/index.html", DIRECTORY, port)
        logger.info("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down.")


if __name__ == "__main__":
    main()
