#!/usr/bin/env python3
"""Check document ownership, local links, and preserved historical text."""

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


REGISTRY = "docs/research/document_registry.json"
STATUSES = {
    "current", "compatibility", "historical", "source_summary",
    "verification_receipt", "user_source_historical",
}
LINKED_STATUSES = STATUSES - {"historical", "user_source_historical"}
BODY_MARKERS = {
    ".md": b"<!-- doc-history-body-begins -->\n",
    ".tex": b"% doc-history-body-begins\n",
}
INLINE_LINK = re.compile(r"\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")
REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE)


def discover_documents(root):
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "--", "*.md", "*.tex"], cwd=root, check=True, capture_output=True,
    )
    return set(result.stdout.decode().strip("\0").split("\0")) - {""}


def check(root):
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    errors = []
    if registry.get("schema_version") != 1:
        errors.append("registry: unsupported schema_version")
    entries = registry["documents"]
    documents = {entry["path"]: entry for entry in entries}
    if len(documents) != len(entries):
        errors.append("registry: duplicate document paths")
    discovered = discover_documents(root)

    def local_path(value, context):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or "\\" in value:
            errors.append(f"{context}: expected repository-relative path: {value}")
            return None
        target = root / path
        if not target.is_file():
            errors.append(f"{context}: missing file: {value}")
            return None
        return target

    def current_reference(value, context):
        local_path(value, context)
        if documents.get(value, {}).get("status") != "current":
            errors.append(f"{context}: reference must name a current owner: {value}")

    for field in ("canonical_entry", "instruction_entry"):
        current_reference(registry[field], field)

    classified = {path: 1 for path in documents}
    raw_count = 0
    for group in registry["raw_groups"]:
        pattern = group["pattern"]
        current_reference(group["owner"], pattern)
        if group.get("status") != "raw_immutable":
            errors.append(f"{pattern}: raw group must be raw_immutable")
        matches = {path for path in discovered if fnmatch.fnmatchcase(path, pattern)}
        snapshots = group["sha256"]
        for path in matches:
            classified[path] = classified.get(path, 0) + 1
        if matches != set(snapshots):
            errors.append(f"{pattern}: snapshot inventory differs: "
                          f"{sorted(matches.symmetric_difference(snapshots))}")
        for path, expected in snapshots.items():
            target = local_path(path, pattern)
            if target and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                errors.append(f"{path}: immutable raw document changed")
        raw_count += len(matches)

    for path in sorted(discovered):
        count = classified.get(path, 0)
        if count != 1:
            errors.append(f"{path}: expected one classification, found {count}")

    for path, entry in documents.items():
        target = local_path(path, "document")
        status = entry["status"]
        if status not in STATUSES:
            errors.append(f"{path}: unknown status: {status}")
        current_reference(entry["owner"], path)
        for replacement in entry.get("superseded_by", []):
            current_reference(replacement, path)
        if not target:
            continue
        content = target.read_bytes()
        if status == "historical":
            prefix = b"% doc-status: historical" if target.suffix == ".tex" else (
                b"<!-- doc-status: historical -->")
            if not content.startswith(prefix):
                errors.append(f"{path}: missing historical banner")
            if not entry.get("superseded_by"):
                errors.append(f"{path}: historical document needs superseded_by")
            marker = BODY_MARKERS[target.suffix]
            parts = content.split(marker, 1)
            if len(parts) != 2:
                errors.append(f"{path}: missing historical body boundary")
            elif hashlib.sha256(parts[1]).hexdigest() != entry["body_sha256"]:
                errors.append(f"{path}: preserved historical body changed")
        if status == "user_source_historical":
            if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                errors.append(f"{path}: preserved user source changed")
        if status not in LINKED_STATUSES:
            continue
        text = content.decode("utf-8")
        if re.search(r"/(?:Users|home)/[^\s`<>]+", text):
            errors.append(f"{path}: machine-specific source path; use a repository locator")
        # Do not interpret example Markdown inside fenced code as real links.
        text = re.sub(r"^(```|~~~).*?^\1[^\n]*$", "", text,
                      flags=re.MULTILINE | re.DOTALL)
        for pattern in (INLINE_LINK, REFERENCE_LINK):
            for match in pattern.finditer(text):
                link = match.group(1).strip("<>")
                parts = urlsplit(link)
                if parts.scheme or parts.netloc or not parts.path:
                    if parts.scheme == "file":
                        errors.append(f"{path}: non-portable local link: {link}")
                    continue
                linked_path = Path(unquote(parts.path))
                destination = (target.parent / linked_path).resolve()
                if linked_path.is_absolute() or (destination != root and root not in destination.parents):
                    errors.append(f"{path}: non-portable local link: {link}")
                elif not destination.exists():
                    errors.append(f"{path}: broken local link: {link}")
    return errors, len(discovered), raw_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors, count, raw_count = check(args.root.resolve())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"FAIL: {len(errors)} document error(s)")
        return 1
    print(f"PASS: {count} documents classified; {raw_count} immutable raw summaries; "
          "historical bodies, owners, and current/source local links verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
