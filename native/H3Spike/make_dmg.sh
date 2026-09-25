#!/bin/sh
# Packages h3c-app.app (already built by package_app.sh) into a compressed,
# distributable .dmg with a drag-to-Applications shortcut - the standard
# way to hand someone else a macOS app. Uses only hdiutil, already part of
# macOS - no third-party dmg-building tool.
set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
app_bundle="$script_dir/.build/h3c-app.app"

if [ ! -d "$app_bundle" ]; then
    echo "h3c-app.app not found at $app_bundle - run ./package_app.sh first" >&2
    exit 1
fi

version=$(defaults read "$app_bundle/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "0.0.0")
dmg_path="$script_dir/.build/h3c-app-$version.dmg"

staging_dir=$(mktemp -d)
trap 'rm -rf "$staging_dir"' EXIT

cp -R "$app_bundle" "$staging_dir/"
ln -s /Applications "$staging_dir/Applications"

rm -f "$dmg_path"
hdiutil create -volname "h3c-app" -srcfolder "$staging_dir" -ov -format UDZO "$dmg_path"

# Signing/notarizing the dmg itself is optional - Gatekeeper's actual check
# happens against the .app's own signature/staple when it's launched from
# the mounted volume, which package_app.sh already handles. This just also
# covers the dmg file itself, for a fully clean AirDrop/download experience.
if [ -n "${H3C_SIGN_IDENTITY:-}" ]; then
    codesign --force --sign "$H3C_SIGN_IDENTITY" "$dmg_path"
    if [ -n "${H3C_NOTARY_PROFILE:-}" ]; then
        xcrun notarytool submit "$dmg_path" --keychain-profile "$H3C_NOTARY_PROFILE" --wait
        xcrun stapler staple "$dmg_path"
    fi
fi

echo "Created $dmg_path"
