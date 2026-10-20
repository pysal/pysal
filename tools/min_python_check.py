#!/usr/bin/env python3
"""
Compute the effective minimum Python version implied by a set of PyPI dependencies.

What it does:
- Takes requirement strings like "libpysal>=4.12.1"
- For each requirement, queries PyPI JSON for:
    - the pinned "floor" version if it's a simple >=X.Y.Z (preferred), else
    - the latest release metadata as a fallback
- Extracts each distribution's declared Requires-Python
- Computes the implied minimum Python as the maximum of those minima

Notes / limitations:
- Accurately handling arbitrary PEP 440 specifier sets (e.g. "!=, ~=, <") is complex.
  This script supports:
    * exact pins: "pkg==1.2.3"
    * floor pins: "pkg>=1.2.3"
  Everything else falls back to querying "latest" metadata and warns you.
- "Requires-Python" can be a range (e.g. ">=3.10,<3.13"). We conservatively take
  the lower bound (>=3.10) for the purpose of "minimum Python".
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple, List


REQS = [
    "libpysal>=4.12.1",
    "access>=1.1.9",
    "esda>=2.6.0",
    "giddy>=2.3.6",
    "inequality>=1.1.1",
    "pointpats>=2.5.1",
    "segregation>=2.5.1",
    "spaghetti>=1.7.6",
    "mgwr>=2.2.1",
    "momepy>=0.9.1",
    "spglm>=1.1.0",
    "spint>=1.0.7",
    "spreg>=1.8.1",
    "tobler>=0.12.1",
    "mapclassify>=2.8.1",
    "splot>=1.1.7",
    "spopt>=0.6.1",
    "gwlearn>=0.1.0",
]


@dataclass(frozen=True)
class Req:
    name: str
    op: Optional[str] = None  # '==', '>=', or None
    version: Optional[str] = None


@dataclass(frozen=True)
class Result:
    name: str
    queried: str  # e.g. "2.6.0" or "latest"
    requires_python: Optional[str]
    min_python: Optional[Tuple[int, int]]  # (major, minor) extracted from lower bound
    warning: Optional[str] = None


def parse_req(req_str: str) -> Req:
    """
    Parse a minimal subset of PEP 508/440 requirement strings:
      - "name>=version"
      - "name==version"
      - "name" (no spec)
    Ignores extras/markers for now.
    """
    s = req_str.strip()
    m = re.match(r"^([A-Za-z0-9_.-]+)\s*(==|>=)?\s*([A-Za-z0-9_.+-]+)?$", s)
    if not m:
        raise ValueError(f"Unparseable requirement: {req_str!r}")
    name, op, ver = m.group(1), m.group(2), m.group(3)
    if op is None:
        ver = None
    return Req(name=name, op=op, version=ver)


def pypi_json(name: str, version: Optional[str]) -> dict:
    """
    Fetch PyPI JSON. If version is None, fetch project JSON (latest).
    """
    if version:
        url = f"https://pypi.org/pypi/{name}/{version}/json"
    else:
        url = f"https://pypi.org/pypi/{name}/json"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "pysal-min-python-check/1.0 (+https://github.com/pysal/)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def extract_min_python(requires_python: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Extract a conservative minimum Python (major, minor) from a Requires-Python spec.

    Examples:
      ">=3.10" -> (3,10)
      ">=3.10,<3.13" -> (3,10)
      "~=3.11" -> (3,11)  (treated as floor)
      None or "" -> None

    If there are multiple lower bounds, we take the largest among them.
    """
    if not requires_python:
        return None

    # Find all lower bounds like ">=3.10" or ">3.10" or "~=3.11"
    bounds: List[Tuple[int, int]] = []
    for m in re.finditer(r"(>=|>|~=)\s*(\d+)\.(\d+)", requires_python):
        maj = int(m.group(2))
        min_ = int(m.group(3))
        bounds.append((maj, min_))

    if not bounds:
        return None

    # Take the max lower bound lexicographically (3,12) > (3,11)
    return max(bounds)


def fmt_py(ver: Optional[Tuple[int, int]]) -> str:
    return f"{ver[0]}.{ver[1]}" if ver else "—"


def main(reqs: List[str]) -> int:
    parsed = [parse_req(r) for r in reqs]

    results: List[Result] = []
    for r in parsed:
        warning = None

        # Prefer querying the pinned floor/exact version.
        query_version: Optional[str] = None
        if r.op in ("==", ">=") and r.version:
            query_version = r.version
        else:
            warning = (
                f"Unsupported specifier '{r.op}{r.version or ''}' "
                f"for {r.name}; querying latest metadata instead."
            )

        try:
            data = pypi_json(r.name, query_version)
            rp = data.get("info", {}).get("requires_python")
            mp = extract_min_python(rp)
            results.append(
                Result(
                    name=r.name,
                    queried=query_version or "latest",
                    requires_python=rp,
                    min_python=mp,
                    warning=warning,
                )
            )
        except Exception as e:
            results.append(
                Result(
                    name=r.name,
                    queried=query_version or "latest",
                    requires_python=None,
                    min_python=None,
                    warning=f"ERROR fetching PyPI JSON: {e}",
                )
            )

    # Compute effective minimum: max of minima we could extract
    minima = [r.min_python for r in results if r.min_python is not None]
    effective = max(minima) if minima else None

    # Print report
    name_w = max(len(r.name) for r in results) if results else 10
    print(
        f"{'Package':{name_w}}  {'Queried':8}  {'Requires-Python':28}  {'MinPy':5}  Notes"
    )
    print("-" * (name_w + 8 + 28 + 5 + 8 + 10))

    for r in sorted(results, key=lambda x: (x.min_python or (0, 0), x.name)):
        notes = r.warning or ""
        rp = r.requires_python or "—"
        if len(rp) > 28:
            rp = rp[:25] + "..."
        print(
            f"{r.name:{name_w}}  {r.queried:8}  {rp:28}  {fmt_py(r.min_python):5}  {notes}"
        )

    print()
    print(f"Effective implied minimum Python (max of minima): {fmt_py(effective)}")

    # Exit nonzero if any fetch failed
    failed = any(r.warning and r.warning.startswith("ERROR") for r in results)
    return 2 if failed else 0


if __name__ == "__main__":
    # Use REQS above by default, or allow passing req strings on CLI
    reqs = sys.argv[1:] if len(sys.argv) > 1 else REQS
    raise SystemExit(main(reqs))
