#!/bin/sh
# Assembles h3c-app.app from a release build of the H3cApp SPM target.
# The Swift target/module itself stays named H3cApp (Swift module names
# can't contain a hyphen), but every user-visible name - the bundle
# folder, the executable inside it, Info.plist - is h3c-app, matching
# the actual product/repo name.
set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
build_dir="$script_dir/.build"
app_bundle="$build_dir/h3c-app.app"
# This repo lives under ~/Documents, which Google Drive's desktop app
# watches/syncs by default; it periodically opens swift build's llbuild
# SQLite database (build.db) for that, and the resulting lock contention is
# what caused the "disk I/O error" below in practice, repeatably - not
# generic llbuild flakiness. Building to a scratch path outside that synced
# tree sidesteps it entirely (verified: reliable across many consecutive
# builds, vs. frequent failures at the in-tree default path).
scratch_dir="${TMPDIR:-/tmp}h3c-app-build-scratch"

# swift build's own llbuild database can still occasionally hit a transient
# "disk I/O error" writing its bookkeeping *after* compiling and linking
# have already succeeded, which still fails the process's exit code - so
# don't trust that exit code, and instead check for a freshly written
# binary below.
rm -f "$scratch_dir/release/H3cApp"
swift build --package-path "$script_dir" --scratch-path "$scratch_dir" -c release || true
if [ ! -x "$scratch_dir/release/H3cApp" ]; then
    echo "swift build did not produce $scratch_dir/release/H3cApp" >&2
    exit 1
fi

rm -rf "$app_bundle"
mkdir -p "$app_bundle/Contents/MacOS" "$app_bundle/Contents/Resources"
cp "$scratch_dir/release/H3cApp" "$app_bundle/Contents/MacOS/h3c-app"
cp "$repo_root/h3_shaders.metal" "$app_bundle/Contents/Resources/h3_shaders.metal"
cp "$script_dir/Packaging/Info.plist" "$app_bundle/Contents/Info.plist"

echo "Packaged $app_bundle"
