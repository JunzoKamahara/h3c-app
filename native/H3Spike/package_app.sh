#!/bin/sh
# Assembles h3c-app.app from a release build of the H3App SPM target.
# The Swift target/module itself stays named H3App (Swift module names
# can't contain a hyphen), but every user-visible name - the bundle
# folder, the executable inside it, Info.plist - is h3c-app, matching
# the actual product/repo name.
set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
build_dir="$script_dir/.build"
app_bundle="$build_dir/h3c-app.app"

# swift build's own llbuild database occasionally hits a transient "disk
# I/O error" writing its bookkeeping *after* compiling and linking have
# already succeeded, which still fails the process's exit code - so
# don't trust that exit code, and instead check for a freshly written
# binary below.
rm -f "$build_dir/release/H3App"
swift build --package-path "$script_dir" -c release || true
if [ ! -x "$build_dir/release/H3App" ]; then
    echo "swift build did not produce $build_dir/release/H3App" >&2
    exit 1
fi

rm -rf "$app_bundle"
mkdir -p "$app_bundle/Contents/MacOS" "$app_bundle/Contents/Resources"
cp "$build_dir/release/H3App" "$app_bundle/Contents/MacOS/h3c-app"
cp "$repo_root/h3_shaders.metal" "$app_bundle/Contents/Resources/h3_shaders.metal"
cp "$script_dir/Packaging/Info.plist" "$app_bundle/Contents/Info.plist"

echo "Packaged $app_bundle"
