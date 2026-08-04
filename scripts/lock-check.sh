#!/usr/bin/env bash
# Do the committed lockfiles still agree with the committed pins?
#
# Deliberately does NOT re-resolve refs. Node refs float (ref = "main"), so
# re-resolving would report drift whenever any upstream repo moved -- a gate
# that fails for reasons unrelated to the commit under test is a gate people
# learn to ignore. `just lock` is how you intentionally pick up upstream
# movement; this only asks whether nodes.lock.json and env/*/requirements.in
# describe the same world.
set -euo pipefail

cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cp -R env "$tmp/env.orig"

rc=0
for backend in cuda rocm; do
    python3 scripts/pin.py collect "$backend" >/dev/null 2>&1
    if ! diff -u "$tmp/env.orig/$backend/requirements.in" "env/$backend/requirements.in"; then
        echo "STALE: env/$backend/requirements.in disagrees with manifest/nodes.lock.json" >&2
        echo "       run 'just lock' and commit the result" >&2
        rc=1
    fi
done

rm -rf env && cp -R "$tmp/env.orig" env

[ $rc -eq 0 ] && echo "lockfiles agree with the pinned commits"
exit $rc
