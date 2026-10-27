"""Guard test for the zod dependency bump (GHSA-mrhx-c7xc-m3w5).

The frontend depends on zod transitively (via eslint-plugin-react-hooks),
so the version is pinned only in frontend/package-lock.json. This test
ensures the lockfile pins a fixed version of zod so the advisory's
vulnerable range is not reintroduced.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCKFILE = REPO_ROOT / "frontend" / "package-lock.json"

FIXED_ZOD_VERSION = (4, 6, 6)


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def test_zod_lockfile_pins_fixed_version():
    assert LOCKFILE.exists(), f"missing lockfile: {LOCKFILE}"

    packages = json.loads(LOCKFILE.read_text(encoding="utf-8"))["packages"]
    assert "node_modules/zod" in packages, "zod is not in the frontend lockfile"

    version = packages["node_modules/zod"]["version"]
    assert _version_tuple(version) >= FIXED_ZOD_VERSION, (
        f"zod {version} is vulnerable to GHSA-mrhx-c7xc-m3w5; "
        f"bump the lockfile pin to >= {'.'.join(map(str, FIXED_ZOD_VERSION))}"
    )
