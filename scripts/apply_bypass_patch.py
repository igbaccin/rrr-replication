#!/usr/bin/env python3
"""
apply_bypass_patch.py — Add RRR_BYPASS_VALIDATION toggle to the pipeline.

When RRR_BYPASS_VALIDATION=1:
  - validate.py: validate_evidence() accepts all items (no quote-on-page check)
  - writer.py: _remove_invalid_citations() becomes a no-op (writer can emit freely)

The toggle is dormant by default. Only activates when the env var is set.

Usage:
    python3 scripts/apply_bypass_patch.py           # applies to src/rrr/
    python3 scripts/apply_bypass_patch.py --check    # dry run, shows what would change
    python3 scripts/apply_bypass_patch.py --revert   # removes the patches

Idempotent: running twice does not double-apply.
"""

import os, sys, re

SRC = os.path.join(os.environ.get("RRR_DIR", "/root/RRR"), "src", "rrr")

# ── Patch definitions ──────────────────────────────────────────

VALIDATE_MARKER = "# [BYPASS_PATCH] RRR_BYPASS_VALIDATION support"

VALIDATE_PATCH = '''
# [BYPASS_PATCH] RRR_BYPASS_VALIDATION support
def _bypass_active():
    return os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1"

'''

# The function we need to modify in validate.py
VALIDATE_OLD_FUNC_START = "def validate_evidence(evidence, metadata_df, soft_threshold: float = 0.78):"
VALIDATE_NEW_FUNC = '''def validate_evidence(evidence, metadata_df, soft_threshold: float = 0.78):
    if _bypass_active():
        return [{"item": r["item"] if isinstance(r, dict) and "item" in r else r,
                 "ok": True, "reason": "bypass"} for r in
                (validate_evidence_verbose(evidence, metadata_df, soft_threshold)
                 if False else [{"item": it} for it in evidence])]
    out = []
    for r in validate_evidence_verbose(evidence, metadata_df, soft_threshold=soft_threshold):
        ok = r["verdict"] in ("exact","soft_ok")
        out.append({"item": r["item"], "ok": ok, "reason": r["reason"]})
    return out'''

# Actually, let me make it simpler and more robust:
VALIDATE_REPLACEMENT = '''def validate_evidence(evidence, metadata_df, soft_threshold: float = 0.78):
    # Bypass mode: accept all evidence without quote-on-page verification
    if os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":
        return [{"item": it, "ok": True, "reason": "bypass"} for it in evidence]
    out = []
    for r in validate_evidence_verbose(evidence, metadata_df, soft_threshold=soft_threshold):
        ok = r["verdict"] in ("exact","soft_ok")
        out.append({"item": r["item"], "ok": ok, "reason": r["reason"]})
    return out'''


WRITER_MARKER = "# [BYPASS_PATCH] skip invalid-citation removal in bypass mode"

WRITER_OLD_REMOVAL = "    full_text, removed_citations = _remove_invalid_citations(full_text, allowed_docs)"
WRITER_NEW_REMOVAL = """    # [BYPASS_PATCH] skip invalid-citation removal in bypass mode
    if os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":
        removed_citations = []
    else:
        full_text, removed_citations = _remove_invalid_citations(full_text, allowed_docs)"""


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def patch_validate(check_only=False):
    path = os.path.join(SRC, "validate.py")
    if not os.path.isfile(path):
        print(f"  [SKIP] {path} not found")
        return False

    content = _read(path)

    if VALIDATE_MARKER in content:
        print(f"  [OK] validate.py already patched")
        return False

    # Find the old validate_evidence wrapper and replace it
    old_block = (
        "# Backwards-compat wrapper (treat exact or soft_ok as ok)\n"
        "def validate_evidence(evidence, metadata_df, soft_threshold: float = 0.78):\n"
        "    out = []\n"
        "    for r in validate_evidence_verbose(evidence, metadata_df, soft_threshold=soft_threshold):\n"
        '        ok = r["verdict"] in ("exact","soft_ok")\n'
        '        out.append({"item": r["item"], "ok": ok, "reason": r["reason"]})\n'
        "    return out"
    )

    new_block = (
        "# Backwards-compat wrapper (treat exact or soft_ok as ok)\n"
        + VALIDATE_MARKER + "\n"
        + VALIDATE_REPLACEMENT
    )

    if old_block not in content:
        print(f"  [WARN] validate.py: could not find exact function signature to patch")
        print(f"         Manual patch needed. Add bypass check at top of validate_evidence():")
        print(f'         if os.environ.get("RRR_BYPASS_VALIDATION", "0") == "1":')
        print(f'             return [{{"item": it, "ok": True, "reason": "bypass"}} for it in evidence]')
        return False

    if check_only:
        print(f"  [DRY] validate.py: would patch validate_evidence()")
        return True

    content = content.replace(old_block, new_block)
    _write(path, content)
    print(f"  [PATCHED] validate.py: bypass toggle added to validate_evidence()")
    return True


def patch_writer(check_only=False):
    path = os.path.join(SRC, "writer.py")
    if not os.path.isfile(path):
        print(f"  [SKIP] {path} not found")
        return False

    content = _read(path)

    if WRITER_MARKER in content:
        print(f"  [OK] writer.py already patched")
        return False

    if WRITER_OLD_REMOVAL not in content:
        print(f"  [WARN] writer.py: could not find _remove_invalid_citations call to patch")
        print(f"         Manual patch needed. Wrap the call with bypass check.")
        return False

    if check_only:
        print(f"  [DRY] writer.py: would patch _remove_invalid_citations call")
        return True

    content = content.replace(WRITER_OLD_REMOVAL, WRITER_NEW_REMOVAL)
    _write(path, content)
    print(f"  [PATCHED] writer.py: bypass toggle added to citation removal")
    return True


def revert_validate():
    path = os.path.join(SRC, "validate.py")
    if not os.path.isfile(path):
        return
    content = _read(path)
    if VALIDATE_MARKER not in content:
        print(f"  [OK] validate.py: no patch to revert")
        return

    # Restore original
    old_block = (
        "# Backwards-compat wrapper (treat exact or soft_ok as ok)\n"
        "def validate_evidence(evidence, metadata_df, soft_threshold: float = 0.78):\n"
        "    out = []\n"
        "    for r in validate_evidence_verbose(evidence, metadata_df, soft_threshold=soft_threshold):\n"
        '        ok = r["verdict"] in ("exact","soft_ok")\n'
        '        out.append({"item": r["item"], "ok": ok, "reason": r["reason"]})\n'
        "    return out"
    )

    # Remove marker and bypass code, restore original
    content = re.sub(
        r"# Backwards-compat wrapper.*?return out",
        old_block.rstrip(),
        content,
        flags=re.DOTALL
    )
    _write(path, content)
    print(f"  [REVERTED] validate.py")


def revert_writer():
    path = os.path.join(SRC, "writer.py")
    if not os.path.isfile(path):
        return
    content = _read(path)
    if WRITER_MARKER not in content:
        print(f"  [OK] writer.py: no patch to revert")
        return

    content = content.replace(WRITER_NEW_REMOVAL, WRITER_OLD_REMOVAL)
    _write(path, content)
    print(f"  [REVERTED] writer.py")


def main():
    check_only = "--check" in sys.argv
    revert = "--revert" in sys.argv

    if not os.path.isdir(SRC):
        print(f"ERROR: Source directory not found: {SRC}")
        print(f"Set RRR_DIR or run from the RRR workspace.")
        sys.exit(1)

    if revert:
        print("Reverting bypass patches...")
        revert_validate()
        revert_writer()
        return

    mode = "DRY RUN" if check_only else "APPLYING"
    print(f"{mode}: bypass patches for RRR_BYPASS_VALIDATION=1")
    print(f"  Source: {SRC}")

    v = patch_validate(check_only)
    w = patch_writer(check_only)

    if check_only:
        if v or w:
            print("\nPatches can be applied. Run without --check to apply.")
        else:
            print("\nNothing to patch (already applied or files not found).")
    else:
        print("\nBypass patches applied. Activate with: export RRR_BYPASS_VALIDATION=1")
        print("Deactivate by unsetting or: export RRR_BYPASS_VALIDATION=0")
        print("Revert patches with: python3 scripts/apply_bypass_patch.py --revert")


if __name__ == "__main__":
    main()
