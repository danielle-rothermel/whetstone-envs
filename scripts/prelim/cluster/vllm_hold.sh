#!/bin/bash
set -uo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: $0 hold|release QROOT PATTERN" >&2
    exit 2
fi
ACTION=$1
QROOT=$2
PATTERN=$3
moved=0
for qdir in "$QROOT"/*/; do
    [[ -d "$qdir/pending" ]] || continue
    mkdir -p "$qdir/hold"
    if [[ "$ACTION" == hold ]]; then
        src="$qdir/pending"
        dst="$qdir/hold"
    else
        src="$qdir/hold"
        dst="$qdir/pending"
    fi
    for f in "$src"/*.json; do
        [[ -e "$f" ]] || continue
        [[ "$(basename "$f")" == *$PATTERN* ]] || continue
        mv "$f" "$dst/" && moved=$((moved + 1))
    done
done
echo "$ACTION $moved jobs matching $PATTERN"
