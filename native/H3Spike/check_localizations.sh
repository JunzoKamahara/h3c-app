#!/bin/sh
# Lists UI strings that have no entry in Packaging/{en,ja}.lproj/
# Localizable.strings. The keys are the Japanese strings in the Swift
# sources: SwiftUI literals (Text("..."), Button("..."), .help("..."), ...)
# and String(localized: "..."). Interpolations become format specifiers
# ("\(n)秒" -> "%lld秒"), so the compiler itself extracts them
# (-emit-localized-strings) rather than a grep. Exits 1 if any are missing.
#
# A string shown in the UI must reach it through one of those two forms;
# a plain String literal is shown as-is and never translated.
set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# A fresh scratch path every time: an incremental build skips unchanged
# files and so would not emit their keys.
swift build --package-path "$script_dir" --scratch-path "$work/build" \
    -c release --target H3cApp \
    -Xswiftc -emit-localized-strings -Xswiftc -emit-localized-strings-path -Xswiftc "$work/keys" >/dev/null

python3 - "$work" "$script_dir/Packaging" <<'EOF'
import glob, json, subprocess, sys
work, packaging = sys.argv[1], sys.argv[2]
keys = {}
for path in sorted(glob.glob(f"{work}/keys/*.stringsdata")):
    data = json.load(open(path))
    source = data["source"].rsplit("/", 1)[-1]
    for entry in data["tables"].get("Localizable", []):
        if entry["key"]:
            keys.setdefault(entry["key"], source)
missing = 0
for lang in ("en", "ja"):
    table = json.loads(subprocess.check_output(
        ["plutil", "-convert", "json", "-o", "-", f"{packaging}/{lang}.lproj/Localizable.strings"]))
    for key, source in keys.items():
        if key not in table:
            print(f"{lang}: missing {key!r} ({source})")
            missing += 1
    for key in table:
        if key not in keys:
            print(f"{lang}: unused {key!r}")
print(f"{len(keys)} keys, {missing} missing")
sys.exit(1 if missing else 0)
EOF
