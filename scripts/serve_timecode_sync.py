"""Serve the timecode sync page locally over HTTP."""
import argparse
import http.server
import socketserver
from functools import partial
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent.parent / "assets" / "timecode_sync"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000).",
    )
    args = parser.parse_args()

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(DIRECTORY))
    with socketserver.TCPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {DIRECTORY} at http://127.0.0.1:{args.port}/index.html")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")


if __name__ == "__main__":
    main()
