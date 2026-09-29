#!/usr/bin/env bash
# Runs the project venv python with an explicit argv[0].
#
# Why: sandboxed executors may launch processes with a foreign argv[0], which
# breaks relocatable (python-build-standalone) interpreters' venv detection.
# Setting argv[0] to the real interpreter path makes the invocation correct in
# any environment. In a normal shell this wrapper is equivalent to running
# ./warden/bin/python directly.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec -a "$DIR/warden/bin/python" "$DIR/warden/bin/python" "$@"
