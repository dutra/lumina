"""Compare the outputs of two runs (or two stage folders) entry by entry and figure by figure.

    python scripts/compare_outputs.py RUN_A RUN_B
    python scripts/compare_outputs.py RUN_A RUN_B --ignore-entry run_summary --ignore-key 'computed_at'
    python scripts/compare_outputs.py RUN_A/final RUN_B/final --dpi 150 --only-figures

RUN_A and RUN_B are run folders (their ``final`` stage is compared, or ``--stage init``) or stage folders.  Compared:

* every table, record, array bundle and figure record of ``results.h5`` and every entry of ``cache.h5``, by content:
  entries only on one side, and for differing entries what differs (columns, index, keys, shapes, dtypes, JSON value
  types: ``true`` is not ``1``, nor ``1`` ``1.0``) and the largest absolute difference of the numbers;
* the list of files under the stage (``results.h5``, ``cache.h5``, their locks, ``logs/`` and the files matching
  ``--ignore-file`` aside);
* text files (``*.tex``, ``*.txt``, ``*.json``, ``*.csv``) line by line, FITS files by their data and header cards,
  other files byte by byte;
* every figure PDF rasterized with ``pdftoppm`` at ``--dpi`` and compared as pixel arrays (max and mean absolute
  difference per figure, in 8-bit levels).

Keys matching ``--ignore-key`` (shell patterns, at any depth of a record, as table columns, array keys and FITS cards)
are left out.  By default so are the record keys named exactly ``seconds``, ``elapsed_seconds``, ``started_at`` and
``computed_at``, at any depth of a record or figure record and nowhere else, so a table column or an array such as
``exposure_seconds`` is always compared (``--no-default-ignores`` keeps them too).
The exit code is 1 when anything differs beyond the ignored entries and keys, 0 otherwise.  Uses only the
environment's packages (numpy, pandas, h5py, astropy) and ``pdftoppm``.
"""
from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_IGNORED_KEYS = ("seconds", "elapsed_seconds", "started_at", "computed_at")    # exact record keys: wall times
SKIPPED_FILES = ("results.h5", "cache.h5", "*.lock", "*.tmp-*", "*.unreadable-*", "logs/*")
TEXT_SUFFIXES = {".tex", ".txt", ".json", ".csv", ".md", ".yaml"}


def stage_of(location: Path, stage: str) -> Path:
    """The stage folder: ``location`` itself when it holds a ``results.h5``, else ``location/stage``."""
    location = Path(location)
    if (location / "results.h5").exists() or not (location / stage).is_dir():
        return location
    return location / stage


class Report:
    """The differences found, printed as they come."""

    def __init__(self):
        self.count = 0

    def __call__(self, kind: str, name: str, text: str) -> None:
        self.count += 1
        print(f"DIFF  {kind:<13} {name}: {text}", flush=True)


def ignored(key: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(str(key), pattern) for pattern in patterns)


def number_difference(a, b) -> str:
    """The largest absolute difference of two numeric arrays of one shape, NaN at the same places counting as equal."""
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind == "b" or b.dtype.kind == "b":
        return f"{int(np.sum(a != b))} of {a.size} values differ"
    a, b = a.astype(float), b.astype(float)
    nan = np.isnan(a) != np.isnan(b)
    both = ~np.isnan(a) & ~np.isnan(b)
    inf = both & (np.isinf(a) | np.isinf(b)) & (a != b)
    finite = both & ~(np.isinf(a) | np.isinf(b))
    delta = np.abs(a[finite] - b[finite])
    differing = int(np.sum(delta > 0) + nan.sum() + inf.sum())
    return (f"{differing} of {a.size} values differ, max abs {delta.max() if delta.size else 0.0:.3g}"
            + (f", NaN at {int(nan.sum())} other places" if nan.any() else ""))


def arrays_differ(a, b) -> str | None:
    """None when the two arrays are equal (NaN equal to NaN), else what differs."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return f"shape {a.shape} != {b.shape}"
    if a.dtype != b.dtype and not (a.dtype.kind in "fiu" and b.dtype.kind in "fiu"):
        return f"dtype {a.dtype} != {b.dtype}"
    if a.dtype.kind in "fiucb" and b.dtype.kind in "fiucb":
        if a.dtype.kind == "c":
            a, b = np.stack((a.real, a.imag)), np.stack((b.real, b.imag))
        same = np.array_equal(a, b, equal_nan=a.dtype.kind == "f" or b.dtype.kind == "f")
        text = None if same else number_difference(a, b)
        if text is None and a.dtype != b.dtype:
            return f"dtype {a.dtype} != {b.dtype} (equal values)"
        return text
    equal = np.array([x == y or (x != x and y != y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist())])
    return None if equal.all() else f"{int((~equal).sum())} of {a.size} values differ"


def compare_json(a, b, path: str, patterns, out: list[str], exact=()) -> None:
    """Append to ``out`` every place where the JSON values ``a`` and ``b`` differ, type included (``True`` is not
    ``1``, nor ``1`` ``1.0``); keys matching ``patterns`` or equal to one of ``exact`` are left out."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if ignored(key, patterns) or str(key) in exact:
                continue
            where = f"{path}.{key}" if path else str(key)
            if key not in a or key not in b:
                out.append(f"{where} only in {'B' if key not in a else 'A'}")
            else:
                compare_json(a[key], b[key], where, patterns, out, exact)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} != {len(b)} items")
            return
        for index, (x, y) in enumerate(zip(a, b)):
            compare_json(x, y, f"{path}[{index}]", patterns, out, exact)
        return
    if type(a) is not type(b):
        out.append(f"{path}: {type(a).__name__} {str(a)[:60]!r} != {type(b).__name__} {str(b)[:60]!r}")
        return
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        if a != b and not (a != a and b != b):
            out.append(f"{path}: {a!r} != {b!r} (abs {abs(a - b):.3g})")
        return
    if a != b:
        out.append(f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}")


def compare_tables(a, b, patterns) -> list[str]:
    out = []
    columns_a = [c for c in a.columns if not ignored(c, patterns)]
    columns_b = [c for c in b.columns if not ignored(c, patterns)]
    if columns_a != columns_b:
        only_a, only_b = set(columns_a) - set(columns_b), set(columns_b) - set(columns_a)
        out.append(f"columns differ (only in A {sorted(map(str, only_a))}, only in B {sorted(map(str, only_b))}, "
                   f"order {'differs' if not (only_a or only_b) else 'n/a'})")
    if len(a) != len(b):
        return out + [f"{len(a)} != {len(b)} rows"]
    if list(a.index.names) != list(b.index.names):
        out.append(f"index names {list(a.index.names)} != {list(b.index.names)}")
    if type(a.index) is not type(b.index) or a.index.dtype != b.index.dtype:
        out.append(f"index {type(a.index).__name__} ({a.index.dtype}) != {type(b.index).__name__} ({b.index.dtype})")
    else:
        text = arrays_differ(a.index.to_numpy(), b.index.to_numpy())
        if text is not None:
            out.append(f"index: {text}")
    for column in (c for c in columns_a if c in set(columns_b)):
        x, y = a[column], b[column]
        if x.dtype != y.dtype:
            out.append(f"column {column}: dtype {x.dtype} != {y.dtype}")
        text = arrays_differ(x.to_numpy(), y.to_numpy())
        if text is not None:
            out.append(f"column {column}: {text}")
    return out


def compare_arrays(a: dict, b: dict, patterns) -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        if ignored(key, patterns):
            continue
        if key not in a or key not in b:
            out.append(f"{key} only in {'B' if key not in a else 'A'}")
            continue
        text = arrays_differ(a[key], b[key])
        if text is not None:
            out.append(f"{key}: {text}")
    return out


def compare_store(stage_a: Path, stage_b: Path, report: Report, *, entry_patterns, key_patterns, record_keys=()) -> int:
    """Compare every entry of ``results.h5`` and ``cache.h5``; returns the number of entries compared.  ``key_patterns``
    apply to every kind of entry, ``record_keys`` (exact names) to the keys of records and figure records only."""
    from kappaccino.io import store
    compared = 0
    for cache in (False, True):
        label = "cache" if cache else "results"
        listed_a, listed_b = store.entries(stage_a, cache=cache), store.entries(stage_b, cache=cache)
        for kind in store.KINDS:
            names_a, names_b = set(listed_a.get(kind, {})), set(listed_b.get(kind, {}))
            for name in sorted(names_a ^ names_b):
                if not ignored(name, entry_patterns):
                    report(f"{label}/{kind}", name, f"only in {'A' if name in names_a else 'B'}")
            for name in sorted(names_a & names_b):
                if ignored(name, entry_patterns):
                    continue
                compared += 1
                try:
                    if kind == store.TABLES:
                        read = store.read_cache_table if cache else store.read_table
                        problems = compare_tables(read(stage_a, name), read(stage_b, name), key_patterns)
                    elif kind == store.ARRAYS:
                        read = store.read_cache_arrays if cache else store.read_arrays
                        problems = compare_arrays(read(stage_a, name), read(stage_b, name), key_patterns)
                    else:
                        if kind == store.RECORDS:
                            read = store.read_cache_record if cache else store.read_record
                            values = read(stage_a, name), read(stage_b, name)
                        else:
                            values = tuple(store.read_figure_record(stage / "figures" / f"{name}.pdf") for stage in (stage_a, stage_b))
                        problems = []
                        compare_json(*values, "", key_patterns, problems, record_keys)
                except Exception as error:      # noqa: BLE001 - a damaged entry is a difference, not a crash
                    problems = [f"cannot be read: {type(error).__name__}: {error}"]
                for problem in problems[:12]:
                    report(f"{label}/{kind}", name, problem)
                if len(problems) > 12:
                    report(f"{label}/{kind}", name, f"... and {len(problems) - 12} more")
    return compared


def stage_files(stage: Path, patterns) -> set[str]:
    files = set()
    for path in stage.rglob("*"):
        if path.is_file():
            relative = path.relative_to(stage).as_posix()
            if not any(fnmatch.fnmatchcase(relative, p) or fnmatch.fnmatchcase(path.name, p) for p in patterns):
                files.add(relative)
    return files


def rasterize(pdf: Path, directory: Path, dpi: int) -> list[np.ndarray]:
    """The pages of ``pdf`` as (row, column, 3) uint8 arrays, from ``pdftoppm -r dpi``."""
    directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftoppm", "-r", str(dpi), str(pdf), str(directory / "page")], check=True, capture_output=True)
    return [read_ppm(page) for page in sorted(directory.glob("page*.ppm"))]


def read_ppm(path: Path) -> np.ndarray:
    """A binary (P6) 8-bit PPM image."""
    data = path.read_bytes()
    fields, position = [], 0
    while len(fields) < 4:
        while data[position:position + 1].isspace():
            position += 1
        if data[position:position + 1] == b"#":
            position = data.index(b"\n", position) + 1
            continue
        end = position
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[position:end])
        position = end
    if fields[0] != b"P6" or int(fields[3]) != 255:
        raise ValueError(f"{path}: not an 8-bit binary PPM")
    width, height = int(fields[1]), int(fields[2])
    return np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=position + 1).reshape(height, width, 3)


def compare_pdf(a: Path, b: Path, dpi: int, scratch: Path) -> tuple[str | None, str]:
    """(None or what differs, a one-line summary) of two PDFs rasterized at ``dpi``."""
    pages_a, pages_b = rasterize(a, scratch / "a", dpi), rasterize(b, scratch / "b", dpi)
    if len(pages_a) != len(pages_b):
        return f"{len(pages_a)} != {len(pages_b)} pages", ""
    worst, mean, pixels = 0, 0.0, 0
    for x, y in zip(pages_a, pages_b):
        if x.shape != y.shape:
            return f"page size {x.shape[1]}x{x.shape[0]} != {y.shape[1]}x{y.shape[0]} pixels", ""
        delta = np.abs(x.astype(np.int16) - y.astype(np.int16))
        worst, mean, pixels = max(worst, int(delta.max())), mean + float(delta.sum()), pixels + delta.size
    summary = f"max abs {worst}, mean abs {mean / max(pixels, 1):.3g} over {len(pages_a)} page(s) at {dpi} dpi"
    return (None if worst == 0 else summary), summary


def compare_fits(a: Path, b: Path, patterns) -> str | None:
    from astropy.io import fits
    with fits.open(a) as ha, fits.open(b) as hb:
        if len(ha) != len(hb):
            return f"{len(ha)} != {len(hb)} HDUs"
        for index, (x, y) in enumerate(zip(ha, hb)):
            if x.data is None or y.data is None:
                if (x.data is None) != (y.data is None):
                    return f"HDU {index}: data in only one"
            else:
                text = arrays_differ(np.asarray(x.data), np.asarray(y.data))
                if text is not None:
                    return f"HDU {index} data: {text}"
            cards_a = {k: v for k, v in x.header.items() if not ignored(k, patterns)}
            cards_b = {k: v for k, v in y.header.items() if not ignored(k, patterns)}
            if cards_a != cards_b:
                keys = sorted(k for k in set(cards_a) | set(cards_b) if cards_a.get(k) != cards_b.get(k))
                return f"HDU {index} header cards differ: {keys[:8]}"
    return None


def compare_files(stage_a: Path, stage_b: Path, report: Report, *, file_patterns, key_patterns, dpi: int,
                  figures: bool, list_figures: bool) -> dict[str, int]:
    files_a, files_b = stage_files(stage_a, file_patterns), stage_files(stage_b, file_patterns)
    for name in sorted(files_a ^ files_b):
        report("file", name, f"only in {'A' if name in files_a else 'B'}")
    counts = {"pdf": 0, "fits": 0, "text": 0, "other": 0}
    scratch = Path(tempfile.mkdtemp(prefix="compare_outputs_"))
    try:
        for name in sorted(files_a & files_b):
            a, b = stage_a / name, stage_b / name
            suffix = a.suffix.lower()
            if suffix == ".pdf":
                if not figures:
                    continue
                counts["pdf"] += 1
                problem, summary = compare_pdf(a, b, dpi, scratch / str(counts["pdf"]))
                if list_figures:
                    print(f"      figure        {name}: {summary or problem}", flush=True)
            elif suffix in (".fits", ".fit"):
                counts["fits"] += 1
                problem = compare_fits(a, b, key_patterns)
            elif suffix in TEXT_SUFFIXES:
                counts["text"] += 1
                lines_a, lines_b = a.read_text(errors="replace").splitlines(), b.read_text(errors="replace").splitlines()
                differing = [i for i, (x, y) in enumerate(zip(lines_a, lines_b)) if x != y]
                problem = None
                if differing or len(lines_a) != len(lines_b):
                    first = differing[0] if differing else min(len(lines_a), len(lines_b))
                    problem = (f"{len(differing)} differing lines, {len(lines_a)} vs {len(lines_b)} lines; first at line {first + 1}: "
                               f"{(lines_a[first] if first < len(lines_a) else '')[:70]!r} vs "
                               f"{(lines_b[first] if first < len(lines_b) else '')[:70]!r}")
            else:
                counts["other"] += 1
                problem = None if a.read_bytes() == b.read_bytes() else "bytes differ"
            if problem is not None:
                report("file", name, problem)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return counts


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    p.add_argument("a", type=Path, help="run folder or stage folder A")
    p.add_argument("b", type=Path, help="run folder or stage folder B")
    p.add_argument("--stage", default="final", help="the stage of a run folder to compare (default final)")
    p.add_argument("--ignore-entry", action="append", default=[], metavar="GLOB", help="store entry names to leave out; repeatable")
    p.add_argument("--ignore-key", action="append", default=[], metavar="GLOB",
                   help="record keys, table columns, array keys and FITS cards to leave out; repeatable")
    p.add_argument("--ignore-file", action="append", default=[], metavar="GLOB", help="stage files to leave out; repeatable")
    p.add_argument("--no-default-ignores", action="store_true",
                   help=f"compare the record keys {', '.join(DEFAULT_IGNORED_KEYS)} too")
    p.add_argument("--dpi", type=int, default=100, help="resolution of the rasterized figures (default 100)")
    p.add_argument("--no-figures", action="store_true", help="do not rasterize the PDFs")
    p.add_argument("--no-store", action="store_true", help="do not compare results.h5 and cache.h5")
    p.add_argument("--list-figures", action="store_true", help="print the pixel difference of every figure, equal or not")
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    stage_a, stage_b = stage_of(args.a, args.stage), stage_of(args.b, args.stage)
    for stage in (stage_a, stage_b):
        if not stage.is_dir():
            print(f"compare_outputs: {stage} is not a folder", file=sys.stderr)
            return 2
    if not args.no_figures and shutil.which("pdftoppm") is None:
        print("compare_outputs: pdftoppm not found (or give --no-figures)", file=sys.stderr)
        return 2
    keys = tuple(args.ignore_key)
    record_keys = () if args.no_default_ignores else DEFAULT_IGNORED_KEYS
    report = Report()
    print(f"A: {stage_a}\nB: {stage_b}", flush=True)
    compared = 0 if args.no_store else compare_store(stage_a, stage_b, report, entry_patterns=args.ignore_entry,
                                                     key_patterns=keys, record_keys=record_keys)
    counts = compare_files(stage_a, stage_b, report, file_patterns=SKIPPED_FILES + tuple(args.ignore_file), key_patterns=keys,
                           dpi=args.dpi, figures=not args.no_figures, list_figures=args.list_figures)
    print(f"compared {compared} store entries, {counts['pdf']} figures, {counts['fits']} FITS, {counts['text']} text and "
          f"{counts['other']} other files: {report.count} difference(s)", flush=True)
    return 1 if report.count else 0


if __name__ == "__main__":
    sys.exit(main())
