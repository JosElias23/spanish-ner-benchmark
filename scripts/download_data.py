"""Download the raw CoNLL-2002 Spanish files and verify their checksums.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spanish_ner.data import sha256  # noqa: E402
from spanish_ner.utils import PROJECT_ROOT, load_config, setup_logging  # noqa: E402

# Mirror of the original CoNLL-2002 distribution (CLiPS, University of Antwerp).
# Same source used by the official Hugging Face loading script.
BASE_URL = "https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/conll2002/"


def main() -> int:
    log = setup_logging()
    config = load_config()
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for filename, expected in config["data"]["checksums"].items():
        dest = raw_dir / filename
        if not dest.exists():
            log.info("Downloading %s ...", filename)
            urllib.request.urlretrieve(BASE_URL + filename, dest)  # noqa: S310
        actual = sha256(dest)
        if actual == expected:
            log.info("%-12s OK  (%s)", filename, actual[:12])
        else:
            log.error("%-12s CHECKSUM MISMATCH\n  expected %s\n  actual   %s",
                      filename, expected, actual)
            failures += 1

    if failures:
        log.error("%d file(s) failed verification.", failures)
        return 1
    log.info("All raw files present and verified in %s", raw_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
