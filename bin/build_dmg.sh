#!/bin/bash
# Build a distributable .dmg from build/Build/Products/Release/whicc.app.
#
# Usage:
#   bin/build_dmg.sh [<version>] [<input.app>] [<output.dmg>]
#
# Defaults:
#   version  <- CFBundleShortVersionString from macui/Info.plist
#   input    <- build/Build/Products/Release/whicc.app
#   output   <- build/dist/whicc-<version>.dmg
#
# Why plain layout (no drag-to-Applications Finder window):
#   That needs a prebuilt .DS_Store + .background PNG + Finder AppleScript.
#   Over-engineering for v0.1.1; plain dmg is still the macOS standard
#   install path (double-click → mount → drag .app to /Applications).
#
# Why UDRW -> mount -> ditto -> convert UDZO:
#   Standard Apple-documented distribution flow. Staging on a writable
#   image lets us place .app + /Applications symlink at the volume root,
#   then convert to compressed read-only for download distribution.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFO_PLIST="$HERE/macui/Info.plist"

# ---- Resolve args ----
VERSION="${1:-}"
INPUT_APP="${2:-$HERE/build/Build/Products/Release/whicc.app}"
OUTPUT_DMG="${3:-}"

if [ -z "$VERSION" ]; then
    [ -f "$INFO_PLIST" ] || { echo "ERROR: Info.plist not found at $INFO_PLIST and no version arg given" >&2; exit 1; }
    VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$INFO_PLIST")
fi

if [ -z "$OUTPUT_DMG" ]; then
    OUTPUT_DIR="$HERE/build/dist"
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DMG="$OUTPUT_DIR/whicc-${VERSION}.dmg"
fi

# ---- Preflight ----
[ -d "$INPUT_APP" ] || {
    echo "ERROR: input .app not found: $INPUT_APP" >&2
    echo "       run 'xcodebuild ... build' first, or pass <input.app> as arg" >&2
    exit 1
}

APP_NAME="$(basename "$INPUT_APP")"
STAGING_DIR="$(mktemp -d -t whicc-dmg)"
RW_DMG="$STAGING_DIR/rw.dmg"

cleanup() {
    [ -n "${STAGING_DIR:-}" ] && [ -d "$STAGING_DIR" ] && {
        [ -d "$STAGING_DIR/mount" ] && hdiutil detach "$STAGING_DIR/mount" 2>/dev/null || true
        rm -rf "$STAGING_DIR"
    }
}
trap cleanup EXIT

echo "==> Input:   $INPUT_APP"
echo "==> Version: $VERSION"
echo "==> Output:  $OUTPUT_DMG"

# ---- Stage volume contents on disk ----
mkdir -p "$STAGING_DIR/staging"
cp -R "$INPUT_APP" "$STAGING_DIR/staging/$APP_NAME"
ln -s /Applications "$STAGING_DIR/staging/Applications"

# ---- Create read-write dmg ----
# -size 700m: whicc.app is ~566 MB (Python venv with mlx/numpy/scipy
# accounts for ~480 MB). 700m leaves ~25% headroom for future growth.
# HFS+ + journaled (-c c=64,a=16,e=16) is the most universally mountable.
# Default format is UDRW when creating from -size alone; UDRW is the
# Apple-recommended staging format before final conversion to UDZO.
echo "==> Creating read-write image (700m)"
hdiutil create \
    -ov \
    -size 700m \
    -fs HFS+ \
    -fsargs "-c c=64,a=16,e=16" \
    -volname "whicc $VERSION" \
    "$RW_DMG"

# ---- Mount + populate ----
echo "==> Mounting read-write image"
mkdir -p "$STAGING_DIR/mount"
hdiutil attach "$RW_DMG" -mountpoint "$STAGING_DIR/mount" -nobrowse

echo "==> Copying contents into mounted volume"
ditto "$STAGING_DIR/staging/$APP_NAME" "$STAGING_DIR/mount/$APP_NAME"
ln -sf /Applications "$STAGING_DIR/mount/Applications"

sync
hdiutil detach "$STAGING_DIR/mount"

# ---- Convert to compressed read-only ----
# -ov: overwrite output if it exists
# -format UDZO: zlib-compressed read-only, standard for download distribution
# -imagekey zlib-level=9: max compression (slower build, smaller download)
echo "==> Converting to UDZO (compressed read-only)"
hdiutil convert "$RW_DMG" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    -o "$OUTPUT_DMG"

# ---- Report ----
SIZE=$(du -h "$OUTPUT_DMG" | cut -f1)
SHA=$(shasum -a 256 "$OUTPUT_DMG" | cut -d' ' -f1)
echo
echo "==> Done: $OUTPUT_DMG"
echo "    size: $SIZE"
echo "    sha256: $SHA"
echo
echo "Install: open the dmg, drag whicc.app to /Applications."
echo "First launch: right-click → Open (Gatekeeper for ad-hoc signed builds)."