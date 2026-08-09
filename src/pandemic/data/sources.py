"""Registry of upstream data sources plus a caching downloader.

Design notes
------------
Every raw file is fetched exactly once and cached on disk under ``data/raw``.
Re-running the pipeline is therefore cheap and works offline. Each download is
recorded in a manifest (URL, SHA-256, byte size, fetch timestamp) so that a
result can always be traced back to the exact bytes that produced it -- the
cheapest possible form of data versioning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import requests

from pandemic.config import RAW, get_logger

log = get_logger(__name__)

MANIFEST = RAW / "manifest.json"
_CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Source:
    """One upstream file."""

    key: str
    url: str
    filename: str
    description: str
    citation: str

    @property
    def path(self) -> Path:
        return RAW / self.filename


SOURCES: dict[str, Source] = {
    "owid": Source(
        key="owid",
        url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv",
        filename="owid-covid-data.csv",
        description=(
            "Country-day panel: cases, deaths, tests, hospital/ICU occupancy, "
            "vaccinations, plus time-invariant covariates (median age, GDP per "
            "capita, population density, hospital beds, HDI)."
        ),
        citation="Our World in Data COVID-19 dataset (Mathieu et al. 2021, Nature Human Behaviour)",
    ),
    "jhu_confirmed": Source(
        key="jhu_confirmed",
        url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv",
        filename="jhu_confirmed_global.csv",
        description="JHU CSSE cumulative confirmed cases, wide by date. Used raw for forensics.",
        citation="Dong, Du & Gardner (2020), Lancet Infectious Diseases 20(5):533-534",
    ),
    "jhu_deaths": Source(
        key="jhu_deaths",
        url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv",
        filename="jhu_deaths_global.csv",
        description="JHU CSSE cumulative deaths, wide by date.",
        citation="Dong, Du & Gardner (2020), Lancet Infectious Diseases 20(5):533-534",
    ),
    "excess_mortality": Source(
        key="excess_mortality",
        url="https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/excess_mortality/excess_mortality.csv",
        filename="excess_mortality.csv",
        description=(
            "All-cause excess deaths (P-scores) from the Human Mortality Database "
            "and World Mortality Dataset. Immune to COVID-attribution bias, so it "
            "is the preferred causal outcome."
        ),
        citation="Karlinsky & Kobak (2021), eLife 10:e69336 (World Mortality Dataset)",
    ),
    "india_states": Source(
        key="india_states",
        url="https://api.covid19india.org/csv/latest/state_wise_daily.csv",
        filename="india_state_wise_daily.csv",
        description="Daily confirmed/recovered/deceased by Indian state, wide format.",
        citation="COVID19-India API (covid19india.org), volunteer-run crowdsourced tracker",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict[str, dict]:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict[str, dict]) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def fetch(key: str, *, force: bool = False, timeout: int = 120) -> Path:
    """Download ``key`` if absent, return its local path, and record provenance.

    Streams to a ``.part`` file and renames on success, so an interrupted run can
    never leave a truncated file that later looks like a valid cache hit.
    """
    src = SOURCES[key]
    if src.path.exists() and not force:
        log.info("cache hit  %-18s %s", key, _human(src.path.stat().st_size))
        return src.path

    log.info("downloading %-17s %s", key, src.url)
    tmp = src.path.with_suffix(src.path.suffix + ".part")
    with requests.get(src.url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
    tmp.replace(src.path)

    manifest = _load_manifest()
    manifest[key] = {
        "url": src.url,
        "filename": src.filename,
        "sha256": _sha256(src.path),
        "bytes": src.path.stat().st_size,
        "fetched_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "citation": src.citation,
    }
    _save_manifest(manifest)
    log.info("saved      %-18s %s", key, _human(src.path.stat().st_size))
    return src.path


def fetch_all(*, force: bool = False) -> dict[str, Path]:
    """Fetch every registered source. Missing optional sources are not fatal."""
    paths: dict[str, Path] = {}
    for key in SOURCES:
        try:
            paths[key] = fetch(key, force=force)
        except Exception as exc:  # noqa: BLE001 - upstream availability varies
            log.warning("could not fetch %s: %s", key, exc)
    return paths


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"
