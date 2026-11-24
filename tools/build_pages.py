"""Stage the GitHub Pages site.

The browser demo runs the library for real, so the site needs the library's
source next to it. Rather than keeping a second copy of `ember/` under `docs/`
(which would drift the first time someone edits an op), this script copies the
tree at build time and serialises it into one JSON file the page can fetch.

Output layout::

    _site/index.html          docs/index.html
    _site/demo_script.py      docs/demo_script.py   (fetched, displayed, executed)
    _site/lib/ember/**.py     a copy of the package
    _site/lib/ember_src.json  {"lib/ember/ops.py": "<source>", ...}

Run locally with ``python tools/build_pages.py``; the Pages workflow runs the
same command, so what you check locally is what gets deployed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
PACKAGE = ROOT / "ember"
OUT = ROOT / "_site"


def copy_static() -> None:
    for name in ("index.html", "demo_script.py"):
        src = DOCS / name
        if not src.is_file():
            raise SystemExit(f"missing {src.relative_to(ROOT)}")
        shutil.copy2(src, OUT / name)


def copy_package() -> list[Path]:
    files = sorted(p for p in PACKAGE.rglob("*.py"))
    if not files:
        raise SystemExit("no Python files found under ember/")
    for src in files:
        dst = OUT / "lib" / src.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return files


def write_manifest(files: list[Path]) -> int:
    manifest = {}
    for src in files:
        key = Path("lib") / src.relative_to(ROOT)
        manifest[key.as_posix()] = src.read_text(encoding="utf-8")
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    target = OUT / "lib" / "ember_src.json"
    target.write_text(payload, encoding="utf-8")
    return len(payload)


def verify(files: list[Path]) -> None:
    """Fail the build instead of deploying a page that cannot run."""
    html = (OUT / "index.html").read_text(encoding="utf-8")
    script = (OUT / "demo_script.py").read_text(encoding="utf-8")
    manifest = json.loads((OUT / "lib" / "ember_src.json").read_text(encoding="utf-8"))

    problems = []
    for key in ("lib/ember/__init__.py", "lib/ember/ops.py", "lib/ember/nn/loss.py"):
        if key not in manifest:
            problems.append(f"manifest is missing {key}")
    for needle, where in (
        ("def train(", "demo_script.py"),
        ("lib/ember_src.json", "index.html"),
        ("demo_script.py", "index.html"),
    ):
        blob = script if where == "demo_script.py" else html
        if needle not in blob:
            problems.append(f"{where} does not reference {needle!r}")
    if problems:
        raise SystemExit("site verification failed:\n  " + "\n  ".join(problems))
    print(f"verified {len(manifest)} modules and a {len(script)}-byte demo script")


def reset_output() -> None:
    if OUT.exists():
        # Some environments refuse to delete a watched directory outright;
        # clearing the contents file by file is enough because the tree shape
        # is rebuilt deterministically below.
        try:
            shutil.rmtree(OUT)
            OUT.mkdir(parents=True)
            return
        except OSError:
            for p in sorted(OUT.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
    OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    reset_output()

    copy_static()
    files = copy_package()
    size = write_manifest(files)
    verify(files)

    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"staged {len(files)} package files + 2 static files")
    print(f"manifest {size:,} bytes as lib/ember_src.json")
    print(f"_site total {total:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
