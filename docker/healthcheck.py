"""Exit 0 if the Flower server accepts TCP connections on its configured port."""

import os
import socket
import sys

port = int(os.environ.get("SERVER_ADDRESS", "0.0.0.0:8080").rsplit(":", 1)[1])

try:
    socket.create_connection(("127.0.0.1", port), timeout=2).close()
except OSError:
    sys.exit(1)