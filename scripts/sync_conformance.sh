#!/usr/bin/env bash
# Sync the shared conformance fixtures from the TypeScript package (StackOneHQ/defender).
#
#   scripts/sync_conformance.sh <ref>   copy conformance/ at <ref> (a tag like defender-v0.9.2, or a commit SHA)
#                                       into tests/conformance/ and record <ref> in tests/conformance/VERSION
#   scripts/sync_conformance.sh --check verify tests/conformance/ matches the ref in VERSION (used in CI)
set -euo pipefail

REPO="StackOneHQ/defender"
DEST="tests/conformance"

# A full commit SHA or a release tag only: a branch would float, and anything else may not be a TS ref.
valid_ref() {
	[[ "$1" =~ ^[0-9a-f]{40}$ || "$1" =~ ^defender-v[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

fetch() {
	local ref="$1" out="$2"
	valid_ref "$ref" || { echo "invalid ref '${ref}': use a full 40-char commit SHA or a defender-vX.Y.Z tag" >&2; exit 2; }
	# Whole archive (small: LFS models come through as pointers); portable across GNU and BSD tar.
	curl -fsSL --retry 3 --retry-delay 2 --max-time 60 "https://codeload.github.com/${REPO}/tar.gz/${ref}" | tar -xz -C "$out" --strip-components=1
	[ -d "$out/conformance" ] || { echo "no conformance/ folder at ${REPO}@${ref}" >&2; exit 1; }
}

cd "$(dirname "$0")/.."
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [ "${1:-}" = "--check" ]; then
	ref="$(tr -d '[:space:]' < "$DEST/VERSION")"
	fetch "$ref" "$tmp"
	if diff -r --exclude VERSION "$tmp/conformance" "$DEST"; then
		echo "tests/conformance matches ${REPO}@${ref}"
	else
		echo "tests/conformance differs from ${REPO}@${ref}: re-run scripts/sync_conformance.sh ${ref}" >&2
		exit 1
	fi
	exit 0
fi

ref="${1:?usage: $0 <ref> | --check}"
fetch "$ref" "$tmp"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$tmp/conformance/." "$DEST/"
echo "$ref" > "$DEST/VERSION"
echo "synced ${REPO}@${ref} → ${DEST}"
