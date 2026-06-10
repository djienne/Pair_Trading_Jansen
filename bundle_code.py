"""Bundle the relevant code and docs of this project into a single Markdown file.

The goal is to produce one self-contained `.md` that can be pasted into a chat,
shared for review, or fed to an LLM. It collects the human-readable source and
documentation (Python, README, config, requirements, the LaTeX transcription,
etc.) and skips binaries, generated output, and the heavy notebooks.

Usage
-----
    python bundle_code.py                  # writes code_bundle.md
    python bundle_code.py -o review.md     # custom output path
    python bundle_code.py --include-notebooks
    python bundle_code.py --max-bytes 200000

By default the file list comes from ``git ls-files`` so only tracked files are
considered and ``.gitignore`` is respected. If git is unavailable the script
falls back to walking the directory tree.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Extensions we treat as bundleable text/code/docs.  Anything not listed here is
# skipped (binaries, images, data dumps, ...).
INCLUDE_EXT = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".toml": "toml",
    ".cfg": "ini",
    ".ini": "ini",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".tex": "latex",
    ".sh": "bash",
    ".ps1": "powershell",
}

# Filenames without a useful extension that we still want to include.
INCLUDE_NAMES = {
    ".gitignore": "text",
    "Dockerfile": "dockerfile",
    "Makefile": "makefile",
}

# Directories to skip entirely (generated output, vcs, caches, envs).
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "node_modules",
    "output",
    "data",
}

# Heavy/binary extensions we always exclude even if they slip through.
ALWAYS_SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".pdf", ".feather", ".parquet", ".csv", ".xlsx", ".pkl",
    ".zip", ".gz", ".tar", ".so", ".dll", ".pyc",
}

# Notebooks are excluded by default (they are large and mostly output cells).
NOTEBOOK_EXT = ".ipynb"


def repo_root() -> Path:
    """Return the git top-level dir, or the script's directory as a fallback."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, check=True,
        )
        return Path(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parent


def list_git_files(root: Path) -> list[Path] | None:
    """List tracked files via git; return None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=root, capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [root / line for line in out.stdout.splitlines() if line.strip()]


def list_walk_files(root: Path) -> list[Path]:
    """Fallback file listing: walk the tree, skipping SKIP_DIRS."""
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def language_for(path: Path) -> str | None:
    """Return the fenced-code language for a file, or None if it should skip."""
    if path.name in INCLUDE_NAMES:
        return INCLUDE_NAMES[path.name]
    return INCLUDE_EXT.get(path.suffix.lower())


def should_include(path: Path, root: Path, include_notebooks: bool,
                   max_bytes: int) -> bool:
    rel_parts = path.relative_to(root).parts
    if any(part in SKIP_DIRS for part in rel_parts):
        return False
    suffix = path.suffix.lower()
    if suffix in ALWAYS_SKIP_EXT:
        return False
    if suffix == NOTEBOOK_EXT:
        return include_notebooks
    if language_for(path) is None:
        return False
    try:
        if path.stat().st_size > max_bytes:
            return False
    except OSError:
        return False
    return True


def read_text(path: Path) -> str | None:
    """Read a file as UTF-8 text; return None if it looks binary."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def build_tree(rel_paths: list[str]) -> str:
    """Render a simple indented file tree from a list of relative paths."""
    lines: list[str] = []
    prev: list[str] = []
    for rel in sorted(rel_paths):
        parts = rel.split("/")
        for depth, part in enumerate(parts[:-1]):
            if depth >= len(prev) or prev[depth] != part:
                lines.append("    " * depth + part + "/")
        lines.append("    " * (len(parts) - 1) + parts[-1])
        prev = parts[:-1]
    return "\n".join(lines)


def fence_for(text: str) -> str:
    """Pick a code fence longer than any backtick run inside the content."""
    longest = 0
    run = 0
    for ch in text:
        if ch == "`":
            run += 1
            longest = max(longest, run)
        else:
            run = 0
    return "`" * max(3, longest + 1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default="code_bundle.md",
                        help="output Markdown file (default: code_bundle.md)")
    parser.add_argument("--include-notebooks", action="store_true",
                        help="include .ipynb notebooks (large; off by default)")
    parser.add_argument("--max-bytes", type=int, default=512_000,
                        help="skip files larger than this many bytes (default: 512000)")
    args = parser.parse_args(argv)

    root = repo_root()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    files = list_git_files(root)
    if files is None:
        files = list_walk_files(root)

    selected: list[Path] = []
    for path in files:
        if not path.is_file():
            continue
        # Never bundle the bundle itself or a previous output.
        if path.resolve() == output_path.resolve():
            continue
        if should_include(path, root, args.include_notebooks, args.max_bytes):
            selected.append(path)

    selected.sort(key=lambda p: p.relative_to(root).as_posix())
    rel_paths = [p.relative_to(root).as_posix() for p in selected]

    if not selected:
        print("No files matched; nothing to bundle.", file=sys.stderr)
        return 1

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out: list[str] = []
    out.append(f"# {root.name} — Code & Docs Bundle\n")
    out.append(
        f"_Generated by `bundle_code.py` on {generated}. "
        f"{len(selected)} files._\n"
    )

    # File tree.
    out.append("## File tree\n")
    out.append("```\n" + build_tree(rel_paths) + "\n```\n")

    # Table of contents with anchor links.
    out.append("## Contents\n")
    for rel in rel_paths:
        anchor = rel.replace("/", "").replace(".", "").replace("_", "").lower()
        out.append(f"- [{rel}](#{anchor})")
    out.append("")

    # File sections.
    for path, rel in zip(selected, rel_paths):
        text = read_text(path)
        if text is None:
            continue
        lang = language_for(path) or "text"
        if path.suffix.lower() == NOTEBOOK_EXT:
            lang = "json"
        out.append(f"\n---\n")
        out.append(f"## {rel}\n")
        fence = fence_for(text)
        out.append(f"{fence}{lang}")
        out.append(text.rstrip("\n"))
        out.append(fence)

    output_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"Wrote {output_path.relative_to(root)} "
          f"({len(selected)} files, {size_kb:.0f} KB).")
    for rel in rel_paths:
        print(f"  + {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
