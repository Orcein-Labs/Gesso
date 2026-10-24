#!/usr/bin/env bash
# Rebuilds vendor/Lava from upstream Lava.jl plus our patch series.
#
#   scripts/vendor_lava.sh [upstream-rev]
#
# With no argument it rebuilds at the rev recorded in vendor/lava-upstream.txt,
# which must reproduce the committed tree byte for byte. To rebase, pass a newer
# upstream rev (or "main"). If a patch no longer applies, the script stops and
# leaves its clone in place: resolve with `git am` there, then re-export the
# series with `git format-patch --zero-commit --no-signature` into
# vendor/patches/lava and run this script again.
#
# LAVA_UPSTREAM_CLONE=<dir> reuses an existing clone instead of fetching.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
record="$here/vendor/lava-upstream.txt"
url="$(awk '$1=="url"{print $2}' "$record")"
rev="${1:-$(awk '$1=="rev"{print $2}' "$record")}"

work="$(mktemp -d)"
if [ -n "${LAVA_UPSTREAM_CLONE:-}" ]; then
    git clone -q "$LAVA_UPSTREAM_CLONE" "$work/lava"
    git -C "$work/lava" fetch -q "$url" 2>/dev/null || true
else
    git clone -q "$url" "$work/lava"
fi
cd "$work/lava"
git -c advice.detachedHead=false checkout -q "$rev"
rev="$(git rev-parse HEAD)"
shopt -s nullglob
patches=("$here"/vendor/patches/lava/*.patch)
if [ ${#patches[@]} -gt 0 ]; then
    if ! git -c user.name=vendor -c user.email=vendor@localhost am -q --committer-date-is-author-date "${patches[@]}"; then
        echo "vendor_lava: patch series does not apply on $rev; clone left at $work/lava" >&2
        exit 1
    fi
fi

rm -rf "$here/vendor/Lava"
mkdir -p "$here/vendor/Lava"
git archive HEAD | tar -x -C "$here/vendor/Lava"
printf 'url %s\nrev %s\n' "$url" "$rev" > "$record"
rm -rf "$work"
echo "vendor/Lava = $url@$rev + ${#patches[@]} patch(es)"
