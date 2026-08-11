# The complete walkthrough

Everything in this project, from the first line to the last, explained in plain
words. No prior knowledge assumed.

If a word looks scary, it is explained the first time it appears. There is also a
[glossary](#glossary) at the very bottom.

---

## Contents

1. [What the project is, in one page](#1-what-the-project-is-in-one-page)
2. [How the folders are arranged](#2-how-the-folders-are-arranged)
3. [The settings file: `config.py`](#3-the-settings-file-configpy)
4. [Getting the data: `data/`](#4-getting-the-data-data)
5. [Part 1, catching bad numbers: `forensics/`](#5-part-1-catching-bad-numbers-forensics)
6. [Part 2, predicting the future: `forecast/`](#6-part-2-predicting-the-future-forecast)
7. [Part 3, finding causes: `causal/`](#7-part-3-finding-causes-causal)
8. [Part 4, hospital beds: `capacity/`](#8-part-4-hospital-beds-capacity)
9. [Drawing the pictures: `viz/`](#9-drawing-the-pictures-viz)
10. [The tests](#10-the-tests)
11. [The runner scripts](#11-the-runner-scripts)
12. [How it all fits together](#12-how-it-all-fits-together)
13. [Glossary](#glossary)

---

# 1. What the project is, in one page

During COVID, every country published numbers every day. How many people got
sick. How many died. How many were in intensive care.

Millions of people made charts from those numbers.

**This project does something different. It does not trust the numbers.**

It asks four questions, and for each one it also asks a second question:
*"how would I know if my answer is wrong?"* Then it actually runs that check and
prints the result, even when the result is embarrassing.

The four questions:

| # | Question | Short answer |
|---|---|---|
| 1 | Can we spot mistakes in the reported data? | Yes, and the popular method for doing it doesn't work |
| 2 | Can we predict what happens next? | A bit, but only when cases are falling |
| 3 | Why did some places do better? | This data cannot tell us, and here is the proof |
| 4 | Which hospitals need more beds? | Yes, with about 13 days of warning |

### A helpful way to think about it

Imagine you are a detective.

A bad detective looks at a crime scene and announces who did it.

A good detective looks at a crime scene, announces who did it, **and then tries
very hard to prove themselves wrong.** They check the alibi. They look for
another explanation. And if the alibi holds up, they say so, out loud, even
though it ruins their story.

This project is the second kind of detective.

---

# 2. How the folders are arranged

```
Pandemic Data Science/
│
├── src/pandemic/          ← All the real code lives here
│   ├── config.py          ← Settings: file paths, constants, random seed
│   ├── data/              ← Downloads and tidies the raw data
│   ├── forensics/         ← Part 1: catching bad numbers
│   ├── forecast/          ← Part 2: predicting the future
│   ├── causal/            ← Part 3: finding causes
│   ├── capacity/          ← Part 4: hospital beds
│   └── viz/               ← Making the charts look good
│
├── scripts/               ← Buttons you press to run each part
├── tests/                 ← 101 checks that the code is correct
├── reports/               ← Where results and charts get saved
│   ├── figures/           ← The PNG charts
│   └── tables/            ← The CSV and JSON result files
├── data/                  ← Downloaded data (not stored on GitHub, too big)
└── .github/workflows/     ← Instructions for GitHub to test the code automatically
```

### Why split code and scripts?

Think of a kitchen.

- `src/pandemic/` is the **recipe book**. It knows how to do things.
- `scripts/` are the **meals**. Each one says "get these ingredients, follow
  these recipes, put the result on this plate."

Keeping them separate means you can test the recipes without cooking a whole
meal. That is why there can be 101 tests that run in under a minute.

---

# 3. The settings file: `config.py`

Every project has numbers that show up in many places. File paths. Constants.
If those are scattered everywhere, changing one means hunting through 30 files.

So they all live in one place.

### Where things are saved

```python
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
```

**Line by line:**

- `Path(__file__)` means "the file I am right now" (`config.py`).
- `.resolve()` turns it into a full address, like
  `D:/Projects/Pandemic Data Science/src/pandemic/config.py`.
- `.parents[2]` means "go up 2 folders". From `src/pandemic/config.py` that
  lands on the project's main folder. So `ROOT` is the project folder, wherever
  it happens to be on your computer.
- `ROOT / "data"` glues `data` onto the end. The `/` here does not mean divide.
  For file paths it means "go into this folder". So `DATA` becomes
  `D:/Projects/Pandemic Data Science/data`.

**Why bother?** Because now the project works on any computer. Nothing is
hard-coded to one machine.

```python
for _p in (RAW, INTERIM, PROCESSED, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)
```

This creates those folders if they do not exist yet.

- `parents=True` means "make any missing parent folders too". If `data/` does not
  exist, make it, then make `data/raw/` inside it.
- `exist_ok=True` means "if the folder is already there, that is fine, do not
  crash".

So a fresh clone of the project builds its own folders on first run.

### The random seed

```python
SEED = 20200311  # WHO pandemic declaration date, as good a seed as any


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
```

**What is a seed?**

Computers cannot make truly random numbers. They follow a recipe that *looks*
random. The seed is the starting point of that recipe.

Same seed → same "random" numbers → same results, every time.

Think of shuffling a deck of cards. If you shuffle by an exact recorded set of
movements, you get the exact same order every time. The seed is that recorded set
of movements.

**Why this matters enormously:** without it, running the project twice gives
slightly different answers, and nobody (including you) could ever check your work.

The number `20200311` is 11 March 2020, the day the WHO declared a pandemic. Any
number would work. This one is easy to remember.

### The disease facts

```python
@dataclass(frozen=True)
class EpiParams:
    """Epidemiological parameters with published uncertainty ranges."""

    # Serial interval, gamma-distributed. Mean 4.7d (95% CrI 3.7-6.0), SD 2.9d.
    # Nishiura, Linton & Akhmetzhanov (2020), Int J Infect Dis 93:284-286.
    serial_interval_mean: float = 4.7
    serial_interval_sd: float = 2.9

    # Case -> hospital admission lag, gamma. Mean ~7d.
    # Docherty et al. (2020), BMJ 369:m1985 (ISARIC-4C cohort).
    onset_to_admission_mean: float = 7.0
    onset_to_admission_sd: float = 4.0

    # Infection-hospitalisation ratio, population-averaged, pre-vaccination.
    ihr_mean: float = 0.030
    ihr_low: float = 0.015
    ihr_high: float = 0.060
```

These are facts about COVID, measured by scientists and published in journals.

- **Serial interval, 4.7 days.** If I catch COVID today and give it to you, on
  average you show symptoms 4.7 days after I did.
- **Admission lag, 7 days.** From testing positive to going into hospital takes
  about a week.
- **IHR = 0.030.** About 3 in every 100 infected people end up in hospital. The
  `low` and `high` versions (1.5% and 6%) say "we are not certain, it is
  somewhere in this range".

**Two things worth noticing.**

First, every number has the paper it came from written next to it. Anyone can
look it up and check I did not invent it.

Second, `frozen=True` means these values **cannot be changed while the program
runs**. If some code tried to secretly modify the serial interval halfway
through, Python would refuse. Constants should stay constant.

### The quantile list

```python
QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
```

A forecast should not be one number. It should be a range.

These seven values describe that range:

- `0.50` is the middle guess. Half the time reality lands above, half below.
- `0.025` and `0.975` are the edges of a 95% range. Reality should land inside
  them 95 out of 100 times.
- The others fill in between.

This exact list is the one the official COVID-19 Forecast Hub used. Copying it
means the scores here can be compared directly to published research.

---

# 4. Getting the data: `data/`

## 4.1 `sources.py`, downloading, once, and remembering

Four data sources, downloaded automatically. Each one is described like this:

```python
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
    ...
}
```

Each source records where it came from and who to credit. Not decoration: if a
result looks strange later, the first question is always "which file did this come
from?"

### The download function

```python
def fetch(key: str, *, force: bool = False, timeout: int = 120) -> Path:
    src = SOURCES[key]
    if src.path.exists() and not force:
        log.info("cache hit  %-18s %s", key, _human(src.path.stat().st_size))
        return src.path
```

**Line by line:**

1. Look up which file we want.
2. If we already downloaded it, **stop right there** and use the copy on disk.

That is called *caching*. The first run downloads 150 MB. Every run after that
takes zero seconds and works with no internet.

```python
    tmp = src.path.with_suffix(src.path.suffix + ".part")
    with requests.get(src.url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                fh.write(chunk)
    tmp.replace(src.path)
```

This is the careful part.

- Download to a **temporary** file ending in `.part`, not the real filename.
- `stream=True` means download in small pieces instead of loading 100 MB into
  memory at once.
- Only when the whole download finishes does `tmp.replace(...)` rename it to the
  real name.

**Why go to the trouble?** Imagine your wifi dies at 80%. Without this, you are
left with a half-file that has the correct name. Next run sees the correct name,
thinks "already downloaded!", and quietly analyses broken data.

With this trick, a failed download leaves a `.part` file that nobody mistakes for
the real thing.

### The fingerprint

```python
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
```

`sha256` produces a long unique code from a file's contents. Change one single
character anywhere in the file and the code changes completely.

Think of it as a fingerprint for data.

This gets saved into `data/raw/manifest.json` along with the URL, the size, and
the exact time of download. Six months from now, if someone asks *"are you sure
this chart used the right data?"*, you can prove it.

## 4.2 `load.py`, tidying the data

### The caching helper

```python
def _cached(name: str, builder: Callable[[], pd.DataFrame], refresh: bool = False) -> pd.DataFrame:
    """Memoise a builder to parquet."""
    path = PROCESSED / f"{name}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = builder()
    df.to_parquet(path, index=False)
    log.info("built %-22s %6d rows x %3d cols", name, len(df), df.shape[1])
    return df
```

The Our World in Data file is a 100 MB spreadsheet. Reading it takes about 8
seconds, which is annoying when you run the pipeline twenty times.

So the first read saves a tidied copy in a fast format called **parquet**. Every
read after that takes a fraction of a second.

`builder` is a *function passed as an argument*. The helper says "if there is no
saved copy, call this function to make one". That way the same caching logic
works for every dataset.

### Removing the fake countries

```python
        # OWID_* codes are aggregates, not countries.
        df = df[~df["iso_code"].astype("string").str.startswith("OWID_", na=True)]
        df = df[df["continent"].notna()]
```

The Our World in Data file contains rows for **"World"**, **"Europe"**,
**"High income countries"**, and so on. They look exactly like countries.

If you leave them in, "World" enters your analysis as a country with 700 million
cases and skews everything.

These fake rows are marked with codes starting `OWID_`, and they have no
continent. Both lines remove them.

This is a two-line fix for a mistake that silently ruins a lot of COVID analyses.

### Why two sources for the same thing

```python
def load_jhu(refresh: bool = False) -> pd.DataFrame:
    """JHU CSSE cumulative counts with *unsmoothed, unclipped* daily differences.

    OWID silently repairs some negative daily values. JHU does not, which makes
    it the right substrate for reporting forensics: a negative ``new_cases`` is a
    downward revision, and those are the events we want to find.
    """
```

Our World in Data quietly fixes obvious errors before publishing. Helpful for
most work.

But Part 1 is **hunting for errors**. You cannot find mistakes in a file where
somebody already cleaned them up. So Part 1 uses the raw Johns Hopkins file
instead.

Small decision. It is the difference between the forensics working and not.

### Turning cumulative into daily

```python
        grp = df.groupby("entity", sort=False)
        df["new_cases"] = grp["cum_cases"].diff()
        df["new_deaths"] = grp["cum_deaths"].diff()
```

Johns Hopkins publishes *running totals*: 100 cases, then 150, then 220.

We want *daily new*: 50, then 70.

- `groupby("entity")` means "handle each country separately". Without it, the
  last day of Afghanistan would be subtracted from the first day of Albania.
- `.diff()` means "today minus yesterday".

Notice what is **not** here: no clipping, no fixing negatives. If a country's
running total goes *down*, `new_cases` comes out negative and we keep it. That
negative is impossible in reality, which makes it exactly the evidence Part 1 is
looking for.

---

# 5. Part 1, catching bad numbers: `forensics/`

## The problem with the obvious method

The standard way to find odd numbers is a **z-score**. Take all the daily case
numbers. Work out the average. Work out how spread out they are. Flag any day
that sits far above the average.

Here is why that fails.

> Imagine measuring someone's heart rate through a whole marathon. Near the
> finish line their heart rate is way above their average for the day. A z-score
> flags those minutes as "unusual".
>
> But nothing is wrong. They were running hard. The number is **correct**.

COVID works the same way. During India's Delta wave, 400,000 cases a day is
enormously above average. That is not an error. That is the pandemic.

So the z-score finds **the tops of the waves**. It is a slow, confusing way to
rediscover the epidemic.

## 5.1 `naive.py`, measuring exactly how badly it fails

Rather than just claiming the z-score is bad, this file measures it.

```python
def naive_zscore_flags(new_cases: pd.Series, threshold: float = 3.0) -> pd.Series:
    x = new_cases.astype(float)
    mu, sd = x.mean(), x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(False, index=x.index)
    return ((x - mu) / sd) > threshold
```

`mu` is the average. `sd` is the spread. `(x - mu) / sd` asks "how many spreads
above average is this day?" Anything above 3 gets flagged.

That is the textbook method, written honestly, so it can be tested.

### The decomposition

Now the interesting part. Every flagged day gets split into three pieces:

```python
    logx = np.log1p(x.clip(lower=0))

    trend = logx.rolling(7, center=True, min_periods=7).mean()
    dev = logx - trend
    dow = pd.Series(x.index.dayofweek, index=x.index)

    wk_effect = dev.groupby(dow).transform("mean")
    resid = dev - wk_effect
```

**Line by line:**

1. `np.log1p(x)` means "take the logarithm of (x + 1)".

   Why? Because epidemics **multiply**. Cases go 100 → 200 → 400 → 800. On a
   log scale, that same growth becomes 1 → 2 → 3 → 4, a straight line. Logs turn
   multiplying into adding, which makes everything easier.

   The "+1" is there because the log of 0 is undefined, and case counts hit 0.

2. `trend` is the average of the 7 days centred on each day (3 before, the day
   itself, 3 after). Using exactly 7 days is deliberate: it averages over one
   complete week, so the weekly rhythm cancels out perfectly.

3. `dev` (deviation) is how far today sits from that local level.

4. `dow` is the day of the week, 0 for Monday through 6 for Sunday.

5. `wk_effect` groups all Mondays together, all Tuesdays together, and so on,
   and takes the average deviation for each. That is the country's normal weekly
   pattern.

6. `resid` (residual) is what is left after removing the trend and the weekday
   pattern. **Only this last piece can be a genuine surprise.**

So:

```
today = the epidemic level  +  the weekly office pattern  +  a surprise
```

### Deciding the verdict

```python
        if date in dump_dates:
            mechanism = "dump"
        elif np.isfinite(d_resid_z) and abs(d_resid_z) > residual_threshold:
            mechanism = "residual"
        elif np.isfinite(d_wk) and abs(d_wk) > np.log(weekday_tolerance):
            mechanism = "weekday"
        else:
            mechanism = "trend"
```

Each flagged day is checked against explanations in order, and gets labelled by
the first one that fits:

| Label | Meaning | Is it a real anomaly? |
|---|---|---|
| `dump` | A batch of withheld cases released at once | **Yes** |
| `residual` | Genuinely unexplained jump | **Yes** |
| `weekday` | Just this country's normal weekly rhythm | No |
| `trend` | The epidemic was already this high | No |

### The result

Across India, the UK, Brazil and the US:

- z-score flags **112 days**
- **111** get labelled `trend` or `weekday`
- **1** survives

Meanwhile the forensic detectors find **43 real reporting events** in those same
countries, and the z-score caught **none** of them.

Roughly **1% precision, 0% recall**. The two methods look at completely
different days.

## 5.2 `digits.py`, do the numbers look hand-made?

Two clever tests that need no knowledge of COVID at all.

### Benford's law (leading digits)

```python
BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))
```

Strange fact: in numbers that grow naturally, the **first digit** is not evenly
spread. About 30% start with 1. Only about 5% start with 9.

Why? Think about a number growing from 100. It has to pass through
100–199 (all starting with 1) before reaching 200. And 100–199 is a wide stretch.
But going from 900 to 1000 is quick. So numbers spend more time starting with 1.

Numbers people invent do not do this. Made-up digits come out too even.

```python
    lead = (v / np.power(10.0, np.floor(np.log10(v)))).astype(int)
```

This extracts the first digit.

- `np.log10(4873)` ≈ 3.68
- `np.floor(3.68)` = 3
- `10 ** 3` = 1000
- `4873 / 1000` = 4.873
- `.astype(int)` chops off the decimals → **4**

```python
    observed = np.bincount(lead, minlength=10)[1:10].astype(float)
    expected = BENFORD_P * observed.sum()
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    p = float(stats.chi2.sf(chi2, df=8))
```

- `bincount` counts how many 1s, 2s, 3s and so on.
- `expected` is what Benford predicts.
- `chi2` measures total disagreement. For each digit: take the gap, square it (so
  too-high and too-low both count as bad), divide by expected (so a gap of 10
  matters more when you only expected 20 than when you expected 2000). Add up.
- `p` converts that into a probability: *if the data were perfectly normal, how
  often would we see a gap this big by pure luck?* Small p means "very rarely, so
  something is going on".

### Last digits (the stronger test)

```python
    last = np.mod(v.astype(np.int64), 10)
    observed = np.bincount(last, minlength=10).astype(float)
    expected = np.full(10, observed.sum() / 10.0)
```

Take any big count, like 4,873. The **last** digit carries no information at all.
Across thousands of days each digit 0–9 should turn up about 10% of the time.

`np.mod(x, 10)` gives the remainder after dividing by 10, which is the last digit.

If the digits are not even, somebody rounded, estimated, or typed the number
instead of counting it.

**This test is stronger than Benford** because it assumes almost nothing. Benford
needs the numbers to span several orders of magnitude. The last-digit test only
needs them to be big.

**Egypt fails this badly**, at p = 6.8e-22. That is 0.00000000000000000000068.
It does not prove anyone lied. It proves the numbers were not produced by
straightforward counting.

## 5.3 `flags.py`, the structural detectors

### The weekly rhythm

```python
    x = new_cases.astype(float)
    centred = x.rolling(7, center=True, min_periods=7).mean()
    ratio = (x / centred).replace([np.inf, -np.inf], np.nan)
```

Divide each day by the average of the week around it.

If a country reports smoothly, every day comes out near **1.0**. Anything else
is the reporting schedule, not the virus.

```python
    def amplitude_of(vals: np.ndarray) -> tuple[np.ndarray, float]:
        total = np.bincount(dow, weights=vals, minlength=7)
        count = np.bincount(dow, minlength=7)
        mult = np.divide(total, count, out=np.ones(7), where=count > 0)
        return mult, float(mult.max() - mult.min())
```

Average those ratios by weekday. That gives seven numbers, one per day. The
**amplitude** is the biggest minus the smallest: how lopsided the week is.

Real results:

| Country | Mon–Sun | What it means |
|---|---|---|
| **Nicaragua** | `0, 0, 6.04, 0.94, 0.03, 0, 0` | Published **once a week**, on Wednesdays |
| **Spain** | Sat `0.19`, Sun `0.11` | Weekend simply not counted |
| **Tanzania** | falls to zero, then stops | Matches the period the state denied the epidemic |

### Proving the pattern is not luck

```python
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations)
    for i in range(n_permutations):
        null[i] = amplitude_of(rng.permutation(r))[1]
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
```

This is a **permutation test**, and it is a genuinely lovely idea.

Shuffle the day labels randomly. Monday's numbers become Thursday's, and so on.
Now any weekly pattern is destroyed by definition. Measure the amplitude anyway.

Do that 500 times. You now have 500 amplitudes from a world where no weekly
pattern exists.

Then ask: **how often did random shuffling beat the real thing?** If the answer
is "almost never", the real pattern is real.

The `+1` on both top and bottom is a small honesty rule. Without it you could
report p = 0, which claims "impossible by chance". With only 500 shuffles you
cannot know that. The `+1` caps the smallest claim you can make at about 0.002.

**Why do it this way instead of a standard statistics test?** Standard tests
assume the data has a nice bell-curve shape. These ratios do not. A permutation
test assumes nothing at all. It just shuffles.

### Backlog dumps

```python
    med = x.rolling(window, center=True, min_periods=window // 2).median()
    is_spike = (x > floor) & (med > 0) & (x / med.replace(0, np.nan) > ratio)
```

A dump is one enormous day after several quiet ones.

Note it uses the **median**, not the average. The median is the middle value when
sorted. It ignores extremes. That matters here, because the giant spike we are
hunting would drag an average upward and hide itself.

```python
        excess = values[pos] - medians[pos]
        deficit = float(np.nansum(np.clip(prior_med - prior, 0, None)))
        rows.append({
            ...
            "conservation": deficit / excess if excess > 0 else np.nan,
        })
    out["is_dump"] = out["conservation"] >= 0.25
```

Here is the clever bit. A spike alone is not enough. The code requires **two**
things:

1. The day towers over its neighbours, **and**
2. The days *before* it were unusually low.

`excess` is how much the spike is above normal. `deficit` is how much the earlier
days were *below* normal. If they roughly match, the cases were being held back
and then released. Mass was conserved.

**Why require both?** Because in a real explosive outbreak the surrounding days
are *also high*. In a paperwork dump, the earlier days are *low*. That second
condition is what tells them apart.

### Frozen series

```python
    change = np.ones(len(vals), dtype=bool)
    change[1:] = vals[1:] != vals[:-1]
    run_id = np.cumsum(change)
```

This finds runs of identical values.

- `vals[1:] != vals[:-1]` compares every day to the previous one. `True` means
  "different from yesterday".
- `np.cumsum` adds them up going along. Every time the value changes, the running
  total ticks up by one. So all days in one unbroken run share the same `run_id`.

Two problems share this signature:

- Runs of **zeros** → reporting stopped
- Runs of the **same non-zero number** → somebody filled a gap with a guess

```python
    agg = agg[(agg["length"] >= min_run) & (agg["active_share"] > 0.5)]
```

`active_share` is the safety catch. Zeros *before* an epidemic starts are
completely normal, not a failure. This only counts runs that happen while the
epidemic is clearly running.

## 5.4 `scorecard.py`, one number per country

Seven detectors get folded into a single 0–100 score.

```python
COMPONENTS: dict[str, tuple[float, float, str]] = {
    "neg_rate":       (0.02, 0.20, "impossible negative daily counts"),
    "gap_severity":   (14.0, 0.15, "reporting stopped mid-epidemic"),
    "fill_severity":  (14.0, 0.15, "series padded with a constant"),
    "dump_rate":      (6.00, 0.15, "batch releases of withheld cases"),
    "weekday_amp":    (1.00, 0.15, "cases reported in weekday batches"),
    "heaping":        (0.50, 0.15, "counts rounded to 0 or 5"),
    "benford_mad":    (0.03, 0.05, "leading digits depart from Benford"),
}
```

Each row is `(cap, weight, description)`.

- **cap** is "this is as bad as it needs to get". Beyond that, worse does not
  count as worse.
- **weight** is how much this detector counts toward the final score. They add
  to 1.0.

Notice Benford gets **0.05**, the smallest weight. That is deliberate. Benford is
the weakest evidence, so it gets the smallest vote.

```python
def _penalty(value: float, cap: float) -> float:
    """Scale a raw metric into [0, 1], saturating at ``cap``."""
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value / cap, 0.0, 1.0))
```

`np.clip(v, 0, 1)` forces the result between 0 and 1.

**Why cap at all?** Suppose one country has a monstrous weekday amplitude of 50.
Without a cap it would dominate the score completely and the other six detectors
would stop mattering. Capping keeps every detector's vote worth what it was meant
to be worth.

### Testing my own judgement

The weights are my opinion. So the code checks whether the ranking actually
depends on them:

```python
    for i in range(n_draws):
        w = rng.dirichlet(concentration * w0)
        ...
        correlations[i] = stats.spearmanr(base, score).statistic
```

A **Dirichlet** draw produces a random set of weights that still add to 1,
centred on my chosen values. Do that 500 times, rebuild the ranking each time,
and compare it to the published one.

`spearmanr` measures how similar two rankings are. 1.0 is identical, 0 is
unrelated.

**Result: median 0.958.** The ranking barely moves. So the ordering is coming
from the data, not from me.

Very few projects check this. It converts "trust my weights" into "here is the
evidence you do not need to".

### The honest hole

```python
        if stats_["active_days"] < min_active_days:
            # Not a pass: a country that barely reported cannot be *scored*, but
            # that silence is itself a finding, so it is recorded rather than
            # dropped.
            excluded.append({...})
            continue
```

**17 countries could not be scored at all** because they reported too little.
Tanzania and Nicaragua are among them.

So the countries with the *worst* data are the ones the index cannot rank. That
is a real weakness. Rather than quietly dropping them, the code keeps the list
and publishes it.

---

# 6. Part 2, predicting the future: `forecast/`

## What we predict, and why

Not tomorrow's raw number. The **7-day average**, 7 and 14 days ahead.

Why? Because Part 1 proved that raw daily numbers are mostly office schedules.
Predicting the raw number means predicting which day a health ministry clears its
paperwork. Real, but useless.

This is a nice moment in the project: **Part 1's finding directly changes Part 2's
design.** The four parts are not four separate mini-projects.

## 6.1 `rt.py`, the epidemiology model

### What Rₜ means

Rₜ is the average number of people each infected person passes the virus to.

- Rₜ = 2 → cases double each round
- Rₜ = 1 → flat
- Rₜ = 0.8 → shrinking

### The serial interval

```python
    shape = ((mean - 1) / sd) ** 2
    scale = sd**2 / (mean - 1)

    k = np.arange(0, max_days + 1, dtype=float)
    w = (k * f(k, shape)
         + (k - 2) * f(k - 2, shape)
         - 2 * (k - 1) * f(k - 1, shape)
         + shape * scale * (2 * f(k - 1, shape + 1)
                            - f(k - 2, shape + 1)
                            - f(k, shape + 1)))
```

This looks horrible. Here is what it is for.

We know the gap between infections averages 4.7 days, spread out in a certain
shape. But our data is daily, so we need a list: *what fraction of transmissions
happen exactly 1 day later? Exactly 2 days? Exactly 3?*

The **obvious** way to convert is wrong, and the docstring says exactly why:

> binning a Gamma CDF at integer edges (`w_s = F(s) - F(s-1)`) yields a
> distribution whose mean is half a day too large, because it assigns the mass of
> `[s-1, s)` to the point `s`

The obvious method quietly adds **half a day** to every gap. That error flows
straight into every Rₜ estimate.

That ugly formula is the published correction from the original paper. There is a
test that checks it comes out at exactly 4.7:

```python
def test_serial_interval_recovers_its_moments(mean, sd):
    """The naive CDF-difference discretisation is half a day too slow; this one is not."""
    w = discretise_serial_interval(mean=mean, sd=sd, max_days=60)
    days = np.arange(w.size)
    got_mean = float(np.sum(w * days))
    assert got_mean == pytest.approx(mean, abs=0.1)
```

```python
    w[0] = 0.0
```

One line, one important idea: `w[0] = 0` means **you cannot infect somebody on
the same day you were infected**. Without it, today's cases would feed into
today's own calculation, and Rₜ would get dragged artificially toward 1.

### Estimating Rₜ

```python
    t = np.arange(tau, n)
    inc_sum = cinc[t + 1] - cinc[t + 1 - tau]
    lam_sum = clam[t + 1] - clam[t + 1 - tau]

    ok = lam_sum > 0
    shape[t[ok]] = prior_shape + inc_sum[ok]
    scale[t[ok]] = 1.0 / (1.0 / prior_scale + lam_sum[ok])

    mean = shape * scale
```

Two ideas here.

**First, a speed trick.** `cinc` is a running total of cases. To get the sum over
any 7-day window, subtract two running totals instead of adding up 7 numbers. That
turns a slow loop into instant arithmetic. On 1,200 days times 40 countries, this
is the difference between minutes and seconds.

**Second, the actual maths.** With a Gamma prior and a Poisson likelihood, the
answer has a closed form. That means no simulation, no sampling, no waiting: two
lines of arithmetic give both the estimate and its uncertainty.

```python
    ok = lam_sum > 0
```

This is honesty in one line. If nobody was infectious yet, Rₜ is **undefined**.
The code leaves it blank rather than quietly returning the prior's guess of 5.
A blank that says "I don't know" beats a confident wrong number.

### Projecting forward

```python
    for k in range(horizon):
        t = n + k
        lo = max(0, t - lag)
        window = buf[lo:t]                       # [I_{t-m}, ..., I_{t-1}]
        lam = float(np.dot(window, w_rev[-window.size:])) if window.size else 0.0
        r_k = 1.0 + (r - 1.0) * (damping**k)
        buf[t] = max(r_k * lam, 0.0)
```

Each new day: look back at recent infections, weight them by how likely each is
to cause a new case today, multiply by Rₜ.

```python
        r_k = 1.0 + (r - 1.0) * (damping**k)
```

**Damping.** With `damping = 0.95`, each day forward pulls Rₜ 5% closer to 1.

This is not a fudge. It encodes something true: neither explosive growth nor
free-fall lasts. People change behaviour. The pool of people who can still catch
it shrinks. Both push Rₜ toward 1.

The backtest includes both the damped and undamped versions so the difference is
measured rather than assumed. **Damped wins.**

## 6.2 `metrics.py`, scoring a forecast honestly

### Why accuracy is not enough

A forecast is not one number. It is "probably 5,000, likely between 3,000 and
8,000".

If you score only how close the middle guess was, you cannot tell apart:

- a model that is right and honest about its uncertainty
- a model that is right but occasionally **wildly overconfident**

The second is dangerous, and plain accuracy will never notice.

### The interval score

```python
def interval_score(y, lower, upper, alpha):
    width = upper - lower
    under = (2.0 / alpha) * np.clip(lower - y, 0, None)
    over = (2.0 / alpha) * np.clip(y - upper, 0, None)
    return width + under + over
```

Three parts, added up. Lower is better.

1. `width`, a penalty just for being vague. A forecast of "between 0 and a
   million" is always right and completely useless.
2. `under`, punishment if reality came in *below* your range.
3. `over`, punishment if reality came in *above* your range.

`np.clip(lower - y, 0, None)` means "if reality was below the bottom of the
range, how far below? Otherwise zero."

The `2/alpha` bit is the important design. For a 95% interval, alpha is 0.05, so
the multiplier is **40**. For a 50% interval, alpha is 0.5, so the multiplier is
**4**.

Meaning: **being wrong when you claimed to be very confident hurts ten times
more.** Exactly right.

### Weighted Interval Score

```python
    total = (0.5 * np.abs(y - median) + sum(w * s for w, s in zip(weights, scores, strict=True))) / denom
```

Combines the middle guess with several ranges (50%, 80%, 95%) into one number.
This is the metric the official COVID-19 Forecast Hubs used, so these scores line
up with published research.

```python
    preds = np.sort(preds, axis=1)
```

Sorting fixes "crossed quantiles", where a model accidentally claims its 25%
value is higher than its 75% value. That is nonsense, and sorting is the standard
repair.

### Geometric mean, not ordinary mean

```python
def relative_skill(scores: np.ndarray, baseline_scores: np.ndarray) -> float:
    """Geometric-mean ratio of model score to baseline score.

    The geometric mean is the right average for a ratio: it is symmetric under
    inversion, so "twice as good" and "twice as bad" are equal and opposite,
    which the arithmetic mean gets wrong.
    """
    ...
    return float(np.exp(np.mean(np.log((s[ok] + eps) / (b[ok] + eps)))))
```

Suppose a model is twice as good on one country and twice as bad on another. It
should come out even.

- Ordinary average of 0.5 and 2.0 → **1.25**. Says "worse". Wrong.
- Geometric mean (multiply, then square root) → **1.0**. Correct.

Taking logs, averaging, then undoing the log is the standard way to compute it.

## 6.3 `models.py`, the six forecasters

### The baseline that must be beaten

```python
class Persistence(Forecaster):
    """"Tomorrow looks like today." The level stays where it is.

    Weak-looking, and remarkably hard to beat once the horizon approaches the
    epidemic's own timescale. Any model that cannot beat this is adding nothing.
    """

    name = "persistence"

    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        return cache.level(i)
```

The whole model is one line: *whatever it is now, it will be that later*.

This is the honest bar. If a complicated model cannot beat this, it is not
earning its complexity. Plenty of published models fail this test and never find
out, because nobody checked.

### The trap this model avoids

```python
class LogLinearDrift(Forecaster):
    """Current exponential growth rate persists, optionally damped.

    Fits ``log1p(7-day average) ~ a + b t`` over the trailing window and
    extrapolates. Epidemics grow multiplicatively, so the log scale is the right
    one. Fitting a polynomial to the *cumulative* curve -- the common shortcut --
    yields a superb in-sample R-squared and no forecasting value at all, because
    a monotone series is trivially fittable and the fit says nothing about the
    increment, which is the quantity anyone actually wants.
    """
```

This docstring names a real trap.

If you fit a curve to the **running total**, you get a beautiful score like
R² = 0.99, because a running total only ever goes up and is trivially easy to
fit. But it tells you nothing about *tomorrow's new cases*, which is the only
thing anyone wants.

This model works on the **daily** numbers, on a log scale.

### The speed trick that makes the honest test affordable

```python
class SeriesCache:
    """Causal features for one entity, precomputed over the full series."""

    def __init__(self, entity, dates, incidence, population):
        ...
        self.avg = trailing_average(self.incidence)
        self.rt = estimate_rt(self.incidence, tau=7).mean
        filled = np.where(np.isfinite(self.avg), self.avg, 0.0)
        self.runmax = np.maximum.accumulate(filled)
```

The backtest makes about 30,000 forecasts. Recalculating Rₜ from scratch for each
would take hours.

But everything used here is **causal**: the value on day 100 depends only on days
1–100. So it can be computed once for a whole country and then just looked up.

That is safe **only** because of the causality. So there is a test that proves it:

```python
def test_rt_estimate_is_causal():
    """R_t at index i must not change when future data is appended or altered."""
    full = estimate_rt(inc, tau=7).mean
    for cut in (120, 200, 260):
        truncated = estimate_rt(inc[:cut], tau=7).mean
        np.testing.assert_allclose(truncated, full[:cut], rtol=1e-9, atol=1e-9)

    perturbed = inc.copy()
    perturbed[200:] *= 7.0
    np.testing.assert_allclose(estimate_rt(perturbed, tau=7).mean[:200],
                               full[:200], rtol=1e-9, atol=1e-9)
```

It multiplies all the **future** data by 7 and checks the past does not move by
even one part in a billion.

If that test ever failed, the entire backtest would be cheating and every result
in Part 2 would be worthless. That is why it exists.

### The machine learning model

```python
class PanelGBM(Forecaster):
    """Gradient-boosted trees trained across all countries at once.

    Predicts the *log growth* from origin to target rather than the level, which
    matters more than the choice of learner: predicting a level forces the model
    to spend capacity re-learning each country's scale, whereas predicting a
    ratio lets every country's wave contribute to one shared question -- given
    this growth pattern, what happens next?
    """
```

The key decision is not "which algorithm". It is **what to predict**.

Predicting the *level* means the model wastes its effort learning that India is
big and Iceland is small. Predicting the *growth ratio* lets every country's wave
answer the same shared question.

```python
        log_growth = float(np.clip(self._model.predict(x)[0], -2.0, 2.0))
```

A safety rail. Tree models cannot extrapolate beyond what they have seen, and on
thin data they can output nonsense. This caps growth at about 7× in either
direction, which is already extreme.

## 6.4 `conformal.py`, honest uncertainty ranges

Every model outputs one number. The range around it is built here, from the
model's **own past mistakes**.

```python
        e = float(np.log1p(max(actual, 0.0)) - np.log1p(max(predicted, 0.0)))
```

Mistakes are measured on the **log scale** because epidemic errors are
multiplicative. "Off by a factor of 2" is meaningful at both 100 cases and
100,000. "Off by 2,000" is meaningless at one end and catastrophic at the other.

```python
def calibrate(point: float, residuals: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Place residual quantiles around a point forecast, on the log scale."""
    if not np.isfinite(point):
        return np.full(levels.size, np.nan)
    offsets = _empirical_quantiles(residuals, levels)
    return np.clip(np.expm1(np.log1p(max(point, 0.0)) + offsets), 0.0, None)
```

"I usually land within 30% either way, so my range is the guess ±30%."

This is called **split conformal prediction**. Its guarantee does not depend on
the errors being bell-shaped, which they very much are not.

### The rule that most projects break

```python
    def add_residual(self, model, entity, horizon, actual, predicted):
```

Residuals are only added when they become **observable**.

If a 14-day forecast is made on the 1st, you do not learn whether it was right
until the 15th. So that mistake cannot be used to tune a forecast made on the 5th.

In `backtest.py`:

```python
            still_pending = []
            for item in pending:
                if item[0] <= origin:
                    _, mname, entity, actual, point = item
                    calibrator.add_residual(mname, entity, horizon, actual, point)
                else:
                    still_pending.append(item)
            pending = still_pending
```

Every mistake sits in a **waiting queue** and is only released when the clock
reaches its target date.

Without this the model is **studying with the answer key**. It looks brilliant in
testing and falls apart in real use.

**The honest result:** the 95% ranges actually contain the truth **85%** of the
time. The 50% ranges hit **43–45%**. Every model is overconfident, and the README
says so.

## 6.5 `backtest.py`, the fair test

```python
    Scale-free aggregation. Country case counts span four orders of magnitude,
    so a raw mean WIS across countries is a ranking of populations. Results are
    aggregated as a geometric-mean ratio against the persistence baseline.
```

India has thousands of times more cases than Iceland, so India's errors are
thousands of times bigger in raw numbers. Averaging them plainly just ranks
countries by size. Comparing each model to the baseline **on the same task**
fixes this.

```python
                if cache.avg[i] < 10:
                    continue
```

Skip stretches where a country had almost no cases. Predicting "2 cases per day"
is not forecasting, it is dividing by noise.

### Splitting by situation

```python
    ratio = df["actual"] / df["baseline_level"].replace(0, np.nan)
    df["regime"] = pd.cut(ratio, [-np.inf, 0.8, 1.25, np.inf],
                          labels=["receding", "flat", "growing"])
```

This is where the project undercuts its own good news.

| Situation | Best model scores |
|---|---|
| Cases **falling** | **0.52** (excellent) |
| Cases **growing** | 0.94 (barely helps) |
| Cases **flat** | worse than doing nothing |

Nearly all the headline 23% improvement comes from predicting **declines**. When
cases are actually **rising**, which is the only time a forecast changes a
decision, the best model beats guessing by about 6%.

Most people would never look. This looks, finds the disappointing answer, and
publishes it.

---

# 7. Part 3, finding causes: `causal/`

This is the hardest part, and the most valuable, because the honest answer turns
out to be **"this data cannot tell us"**, and the code *proves* that instead of
guessing.

## The question

Did stricter lockdowns save lives?

## 7.1 `dag.py`, draw the map before touching the data

Before any analysis, write down what you think causes what.

```python
EDGES: list[tuple[str, str]] = [
    # Demography and wealth drive both the response and the death toll.
    ("median_age", "deaths"),
    ("median_age", "stringency"),
    ("wealth", "stringency"),
    ("wealth", "deaths"),
    ...
    # Post-treatment consequences. Present in the graph precisely so the
    # criterion can refuse them.
    ("stringency", "testing"),
    ("testing", "observed_cases"),
    ("stringency", "vaccination_speed"),
    ("vaccination_speed", "deaths"),
]
```

Each pair `(A, B)` means "A causes B". `("median_age", "deaths")` says an older
population leads to more deaths.

This is a **DAG**: Directed Acyclic Graph.

- *Directed*: arrows point one way. Age causes deaths, not the reverse.
- *Acyclic*: no loops. Nothing can end up causing itself.
- *Graph*: dots connected by arrows.

**Why write it down?** Because a variable list hides your assumptions. A picture
with arrows makes every assumption visible, so someone who disagrees can point at
exactly which arrow is wrong.

### The rule

```python
    Z is admissible for estimating the effect of T on Y if
      (i)  no member of Z is a descendant of T, and
      (ii) Z d-separates T from Y in the graph with all edges out of T removed.
```

This is **Pearl's back-door criterion**, and it decides which variables to adjust
for. Crucially, the code *runs* it. It does not just state an answer.

```python
def is_valid_backdoor(g, treatment, outcome, adjustment):
    """Check Pearl's back-door criterion. Returns (valid, explanation)."""
    if treatment in adjustment or outcome in adjustment:
        return False, "the adjustment set contains the treatment or the outcome"

    descendants = nx.descendants(g, treatment)
    bad = adjustment & descendants
    if bad:
        return False, (f"post-treatment variables in the adjustment set: {sorted(bad)} "
                       "-- these are mediators, and conditioning on them removes part "
                       "of the effect being estimated")

    mutilated = g.copy()
    mutilated.remove_edges_from(list(g.out_edges(treatment)))
    if not _d_separated(mutilated, {treatment}, {outcome}, set(adjustment)):
        return False, "an unblocked back-door path remains between treatment and outcome"

    return True, "valid: blocks every back-door path and contains no descendant of the treatment"
```

**Line by line:**

1. Obviously you cannot adjust for the thing you are measuring.

2. `nx.descendants(g, treatment)` finds everything downstream of lockdown, i.e.
   everything lockdown causes. If any of those are in your adjustment list,
   **reject**.

   > This is the big one. Adjusting for a consequence of lockdown is like asking
   > *"did exercise improve health, ignoring any change in fitness?"* You just
   > deleted the effect you were trying to measure.

   The graph deliberately includes testing, transmission, observed cases and
   vaccination speed **so the rule can refuse them.**

3. `mutilated` is a copy of the graph with all arrows *out of* lockdown deleted.
   Then it checks whether any sneaky path still connects lockdown to deaths.
   Such a path would be a confounder we forgot.

There is a test for each refusal:

```python
@pytest.mark.parametrize("mediator", ["testing", "transmission", "vaccination_speed",
                                      "observed_cases"])
def test_adjusting_for_a_mediator_is_rejected(mediator):
    """Controlling for a consequence of treatment removes part of the effect."""
    ok, why = is_valid_backdoor(g, "stringency", "deaths", base | {mediator})
    assert not ok
    assert "post-treatment" in why
```

## 7.2 `dataset.py`, designing the comparison

The design choices matter far more than the maths that follows.

### Epidemic time, not calendar time

```python
        t0 = base["date_100_cases"]
        ...
        t_end = t0 + pd.Timedelta(days=TREATMENT_WINDOW_DAYS)
        y_end = t0 + pd.Timedelta(days=OUTCOME_WINDOW_DAYS)
```

Every window starts on the day that country hit its **100th case**, not on a
fixed calendar date.

> Comparing Italy in March 2020 with New Zealand in March 2020 compares a country
> three weeks into an outbreak against one that barely had cases. That is a
> difference in *stage*, and it would look like a difference in policy.

Lining everyone up on their own 100th case removes that. It also fixes the case
count at day zero, which kills the most obvious source of backwards reasoning.

### Deaths, not cases

The outcome is deaths, not cases. Why?

> Confirmed cases measure testing effort as much as infection. A country that
> tests ten times more finds more cases at identical true prevalence. Deaths are
> far from clean, but much less elastic to surveillance effort.

### The falsification outcome

```python
        # --- falsification outcome: deaths in the first 21 days.
        # Policy cannot have caused these. Infection to death runs about three
        # weeks, and policy needs one to two more to change infections, so
        # anything dying inside 21 days of t0 was already infected when the
        # window opened. An association here is confounding, by construction.
        early = g[(g["date"] >= t0) & (g["date"] < t0 + pd.Timedelta(days=21))]
```

Clever idea. Deaths within 21 days of the window opening **cannot** have been
prevented by a policy started in that window, because those people were already
infected.

So if lockdown appears to "affect" those deaths, that apparent effect is pure
confounding. It is a built-in lie detector.

## 7.3 `estimators.py`, Double Machine Learning

### Why not just a regression?

> A linear model imposes that every confounder enters additively and linearly.
> Age structure does not affect COVID mortality linearly -- risk rises roughly
> exponentially with age -- so a linear age term leaves residual confounding that
> lands on the treatment coefficient.

An ordinary regression assumes everything works in straight lines. COVID risk vs
age is emphatically not a straight line, so the leftover mess contaminates the
answer.

### The procedure

```python
        for train, test in folds.split(x):
            m_y = clone(base)
            m_t = clone(base)
            m_y.fit(x[train], y[train])
            m_t.fit(x[train], t[train])
            y_hat[test] = m_y.predict(x[test])
            t_hat[test] = m_t.predict(x[test])

        y_res = y - y_hat
        t_res = t - t_hat
        denom = float(np.mean(t_res**2))
        theta = float(np.mean(t_res * y_res) / denom)
```

In plain words:

1. Train a model to predict **deaths** from the background factors (age, wealth,
   density...). Call the leftover `y_res`. That is the part of deaths those
   factors cannot explain.
2. Train another model to predict **lockdown strictness** from the same
   background factors. Leftover is `t_res`.
3. See whether the two leftovers move together.

> Take out everything the background explains. Whatever still lines up afterwards
> is the bit we actually care about.

### Cross-fitting

```python
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
```

`KFold(5)` splits the countries into 5 groups. Predictions for group 1 come from
a model trained on groups 2–5, and so on.

**Why?** A flexible model partly memorises its training data. If it predicted the
same countries it learned from, its leftovers would be artificially small and the
final answer biased. Predicting only unseen countries prevents that.

### Repeating

```python
    thetas_arr = np.asarray(thetas)
    theta = float(np.median(thetas_arr))
    var = float(np.median(np.asarray(variances) + (thetas_arr - theta) ** 2))
```

Which countries land in which fold is random, and that shifts the answer
slightly. So the whole thing runs 20 times with different splits and takes the
middle answer.

The `(thetas_arr - theta) ** 2` term folds the *spread across splits* into the
uncertainty. Without it, a result that was pure luck of one split would look
precise.

### The units problem

```python
    # Rescale to *per stringency point* before returning. The raw ATE answers a
    # different question from the other estimators -- the effect of crossing the
    # median, a jump of roughly 20 points -- and reporting it beside per-point
    # coefficients on one axis silently compares incompatible units.
    contrast = float(t_cont[treated].mean() - t_cont[~treated].mean())
    ate_pp, se_pp = ate / contrast, se / contrast
```

One of the four methods naturally answers a different-sized question. Reporting
it next to the others without converting would be like putting kilometres and
miles in the same column.

## 7.4 `refute.py`, trying to break the answer

Four stress tests, each with a *predicted* outcome if the answer is real.

### Placebo

```python
    for _ in range(n_draws):
        d = df.copy()
        d[treatment] = r.permutation(d[treatment].to_numpy())
        try:
            placebo.append(estimator(d, controls).estimate)
        except Exception:  # noqa: BLE001 - a degenerate draw should not abort the suite
            continue
```

Shuffle the lockdown values randomly between countries. Now the "treatment" is
nonsense. Re-run everything.

**If a real effect exists, it must vanish here.** If shuffled data reproduces
your finding, you were never measuring the treatment.

Result: shuffled average **+0.0009** versus real **+0.0347**. It vanishes. Pass.

### The E-value

```python
    def _ev(rr: float) -> float:
        if rr < 1:
            rr = 1.0 / rr
        return float(rr + np.sqrt(rr * (rr - 1.0)))
```

Asks: *how strong would a hidden factor we never measured have to be, to explain
this away entirely?*

- E-value near 1 → a weak hidden factor is enough. Fragile finding.
- E-value of 4 → the hidden factor would have to be stronger than anything we
  measured. Hard story to tell.

Here it comes out at **2.18**. Moderate.

## 7.5 `identification.py`, the part that matters most

Here is the idea that makes this project serious.

```python
Refutation tests (:mod:`pandemic.causal.refute`) ask whether an estimate is
*stable*. They cannot tell you whether it is *causal*: an estimate driven
entirely by reverse causality is perfectly stable, survives every placebo, and
has a large E-value. Stability and identification are different properties, and
conflating them is how a confounded number acquires a confident standard error.
```

Read that twice. **Passing every robustness check does not mean the answer is
right.**

Those checks measure whether a number is *steady*. They say nothing about
whether it points the right way.

### Measuring the backwards-ness directly

```python
    concurrent = ols_effect(
        sub, outcome=treatment, treatment=concurrent_col,
        controls=[c for c in controls if c != concurrent_col],
        method="stringency ~ outbreak size during the window")
```

Instead of predicting deaths from lockdown, this predicts **lockdown from the
size of the outbreak happening at the time**.

Result: **+2.26, p = 0.032.** Governments clamped down harder where the epidemic
was worse.

> It is like noticing that ambulances are found wherever accidents happen, and
> concluding that ambulances cause accidents.

So countries with strict lockdowns are, by definition, the countries already in
serious trouble. The cause and the effect are decided at the same moment by the
same thing, and no amount of adjusting for fixed country traits separates them.

### The verdict the code writes itself

```python
        verdict = (
            "The design does NOT identify a causal effect. Treatment is assigned in "
            "response to the outbreak, and the association reproduces on a window "
            "the policy could not have affected. The reported coefficient is a "
            "measure of which countries were already in trouble when they acted -- "
            "reporting it as the effect of policy would invert the direction of the "
            "real relationship. Identification requires variation in policy that is "
            "unrelated to local epidemic severity: staggered adoption with matched "
            "pre-trends, a discontinuity, or an instrument."
        )
```

The pipeline prints its own conclusion, including what study design *would* work.

### And it admits its own tests are weak

```python
        "power_caveat": (
            "Both probes are one-sided: they can reveal confounding but cannot "
            "demonstrate its absence. The sign of the estimate is the stronger "
            "evidence here -- a positive coefficient means stricter responses "
            "accompany higher mortality, which is implausible as a causal effect "
            "and expected under simultaneity."
        ),
```

Even the lie detector says "I can catch a lie, but I cannot prove honesty."

## 7.6 `synthetic_control.py`, building a fake country

When one region does something and no single other region is a fair comparison,
**build** one out of a blend of others.

```python
def _fit_weights(y_treated: np.ndarray, y_donors: np.ndarray) -> np.ndarray:
    """Simplex-constrained least squares: min ||y - Dw||, w >= 0, sum(w) = 1."""
    result = minimize(
        loss, w0, jac=grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones_like(w)}],
        options={"maxiter": 800, "ftol": 1e-12},
    )
```

Find the mix of other states whose combined history best matches Maharashtra's,
**before** the lockdown.

Two rules do the real work:

- `bounds=[(0.0, 1.0)]`, no negative amounts. You cannot use "minus 30% of
  Kerala".
- `sum(w) = 1`, the amounts add to exactly 100%.

Together these mean the fake region can only ever be a **weighted average of
things that actually happened**. It can never invent a scenario outside what the
real data contains.

### Testing with no real control group

```python
    all_ratios = np.asarray([real.rmspe_ratio, *ratios.values()])
    rank = int(np.sum(all_ratios >= real.rmspe_ratio))
    p = rank / all_ratios.size
```

With only one treated region there is no sampling variation to appeal to. So
pretend each *donor* was the treated one, run the whole thing again, and see
where the real one ranks.

### The honest failure

```python
    if fake_bigger or not_significant:
        return (
            f"NULL RESULT, and the design does not support interpreting it. ..."
```

For Maharashtra:

- post/pre ratio **0.94** (no jump after the lockdown)
- placebo p-value **1.00** (ranked last of 26)
- a **fake** lockdown 45 days earlier produces a *bigger* apparent effect (1.59)

The chart shows why: the fake Maharashtra stopped matching the real one three
weeks *before* the lockdown. If the comparison already did not match beforehand,
any difference afterwards means nothing.

Reported as **a failed attempt at identification**, not "we found no effect".
Those are different claims.

---

# 8. Part 4, hospital beds: `capacity/`

## 8.1 `convolve.py`, from cases to ICU beds

### The chain

```
cases_t  --(IHR, admission lag)-->  admissions_t
admissions_t  --(critical-care share)-->  ICU admissions_t
ICU admissions  --(length-of-stay survival)-->  ICU census_t
```

### The idea that makes this useful

```python
The last step is the one a regression on cases cannot reproduce. Census is not
proportional to admissions. It is admissions convolved with the probability that
a patient admitted ``s`` days ago is still there. COVID ICU stays are long and
right-skewed (mean ~12 days, SD ~8), so census keeps climbing for one to two
weeks after admissions peak.
```

**ICU occupancy is not the same as ICU admissions.**

> Think of a car park. Cars arriving per hour is one thing. Cars *currently
> parked* is another. If everyone stays 12 hours, the car park keeps filling long
> after arrivals have slowed.

Because COVID ICU stays average about 12 days, **occupancy keeps rising for one
to two weeks after cases have already peaked.**

A simple model that treats ICU as "cases × some number" calls the turning point a
fortnight too early. Which is exactly when the decision to open emergency
capacity has to be made.

### The survival curve

```python
def los_survival(mean: float, sd: float, max_days: int = 60) -> np.ndarray:
    """``S[s] = P(length of stay > s)`` -- the share of admissions still present.

    This is the occupancy kernel. Summing it gives the mean length of stay, which
    is the factor converting a steady admission rate into a steady census.
    """
    ...
    return np.clip(dist.sf(days), 0, 1)
```

`S[s]` = the chance a patient admitted `s` days ago is **still there**.

- `S[0]` = 1.0 (everyone admitted today is still here)
- `S[12]` ≈ 0.5 (about half are still here after 12 days)
- `S[40]` ≈ small

`sf` is the "survival function", the opposite of the usual cumulative
distribution. The area under this curve equals the mean stay, which is the
multiplier that turns a steady arrival rate into a steady occupancy.

### The causal convolution

```python
def _convolve_causal(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution: ``y[t] = sum_s x[t-s] * kernel[s]``, same length as x."""
    x = np.nan_to_num(np.clip(np.asarray(x, float), 0, None))
    full = np.convolve(x, np.asarray(kernel, float), mode="full")
    return full[: x.size]
```

**Convolution** sounds terrifying and is simple: *for today's occupancy, look back
at every past day, weight it by how likely those patients are to still be here,
and add it all up.*

`full[: x.size]` chops off the tail, which keeps the result the same length as
the input and guarantees today never depends on tomorrow.

A test proves it:

```python
def test_convolution_is_causal():
    """Occupancy today cannot depend on cases reported tomorrow."""
    cases = np.zeros(60)
    cases[30] = 1000.0
    census = cases_to_icu_census(cases)
    assert np.all(census[:30] == 0.0)
    assert census[31:].sum() > 0
```

Put 1,000 cases on day 30 and nothing else. Check that days 0–29 stay at exactly
zero.

### The warning time

```python
def peak_lag_days(cases: np.ndarray, census: np.ndarray) -> int:
    ...
    return int(np.argmax(v) - np.argmax(c))
```

`np.argmax` finds *where* the biggest value sits. The gap between the case peak
and the ICU peak is the warning you get.

Measured across 202 countries: **13 days**.

## 8.2 `validate.py`, does it actually work?

About 30 countries published real daily ICU numbers, so this is checkable.

```python
    # Scalar level calibration: least squares through the origin.
    k = float(np.sum(pred[ok] * obs[ok]) / np.sum(pred[ok] ** 2))
```

Only **one** number is fitted per country: an overall scale factor. The *shape*
is pure physics, untouched.

And that one number is interesting in itself:

```python
        "interpretation": (
            "Shape and timing transfer across countries with a fixed parameter set; "
            "the level does not. The spread in fitted multipliers is an estimate of "
            "how differently countries ascertained infections, since the assumed "
            "hospitalisation rate is held common."
        ),
```

Since the hospitalisation rate is held the same everywhere, a country needing a
5× multiplier was finding roughly a fifth of its infections. The **4.5× spread**
across countries is a measurement of how differently countries were detecting
COVID.

### Splitting by era

```python
VACCINE_CUTOFF = "2021-02-01"
POST_VACCINE_START = "2021-07-01"
```

| Period | Countries | Accuracy |
|---|---|---|
| Before vaccines | 28 | **0.938** |
| After vaccines | 38 | 0.565 |

Very accurate, then it breaks.

**And the breaking is correct.** The model assumes a fixed percentage of infected
people need hospital. Vaccines changed that percentage. So the model was always
going to fail there, and it fails exactly on schedule.

> A model that fails for a reason you can name and predict is far more
> trustworthy than one that works for reasons nobody understands.

Pooling both eras gives 0.37, a number that describes neither and makes a working
model look broken.

## 8.3 `risk.py`, turning a forecast into a decision

```python
def _sample_params(r, n: int) -> dict[str, np.ndarray]:
    """Draw epidemiological parameters from their published ranges.

    Triangular distributions over [low, mean, high]: they respect the bounds
    exactly and put most weight on the central estimate, which is the honest
    reading of a literature range without pretending to know its shape.
    """
    return {
        "ihr": r.triangular(EPI.ihr_low, EPI.ihr_mean, EPI.ihr_high, n),
        ...
    }
```

We do not know the exact hospitalisation rate. We know it is somewhere between
1.5% and 6%, most likely around 3%.

A **triangular** distribution says exactly that: never outside the range, most
likely in the middle. Honest about a literature range without pretending to know
more.

```python
    for i in range(n_draws):
        future = project_renewal(hist, float(r_draws[i]), horizon, w=w, damping=damping)
        full = np.concatenate([hist, future])
        out[i] = cases_to_icu_census(full, ihr=..., icu_share=..., ...)
```

Run the whole simulation 300 times with different plausible parameters. That
gives 300 possible futures, which becomes the uncertainty band.

### Two kinds of uncertainty

The docstring separates them, and the distinction is real:

> *Epidemiological parameters.* [...] roughly *multiplicative and persistent* --
> if the true IHR is at the top of its range, it is high for the whole
> projection, so it widens the level without changing the timing.
>
> *Transmission.* [...] dominates the *timing* of a breach.

### The ranking decision

```python
    # Rank by absolute per-capita pressure, not by the ratio to each region's own
    # history. The ratio answers "is this unusual *here*", which sounds like the
    # better question but is dominated by regions that previously had almost no
    # epidemic -- a place going from 0.1 to 2 ICU patients per 100k tops a ratio
    # ranking while needing no help at all.
    return (out.sort_values("projected_peak_per_100k", ascending=False)
            .reset_index(drop=True))
```

I tried the intuitive ranking first (worst compared to your own history) and it
gave nonsense. Sri Lanka came top with a "47× worse than usual" score while
having only 1.75 ICU patients per 100,000.

Ranking by absolute per-capita pressure gives Slovenia, Switzerland, Belgium,
Czechia. All four went on to be among Europe's most strained systems that winter.

### The guard rail

```python
    # A region that never had a wave has a prior peak near zero, and dividing by
    # it turns a small outbreak into an "infinite" utilisation. Below this floor
    # the ratio is not reported at all rather than reported as a huge number:
    # the honest statement is "no prior peak worth comparing to".
    benchmark_usable = np.isfinite(benchmark_per_100k) and benchmark_per_100k >= 1.0
```

When a region has no meaningful history to compare against, the code prints
"no prior peak to compare" rather than a made-up giant number.

---

# 9. Drawing the pictures: `viz/`

```python
@contextlib.contextmanager
def use_theme(mode: str):
    """Apply the theme's rcParams for the duration of the block."""
    p = PALETTES[mode]
    rc = {
        "figure.facecolor": p.surface,
        "axes.spines.top": False,
        "axes.spines.right": False,
        ...
    }
    with mpl.rc_context(rc):
        yield p
```

One place decides how every chart looks. Change it once and all 28 charts update.

`axes.spines.top: False` removes the box lines at the top and right of each
chart. Less ink, same information.

```python
def render(name: str, draw, *, figsize=(10.0, 5.5), modes=MODES) -> list[Path]:
    """Draw one figure in every theme mode and write it to ``reports/figures``."""
    paths = []
    for mode in modes:
        with use_theme(mode) as p:
            fig = plt.figure(figsize=figsize)
            try:
                draw(fig, p)
                out = FIGURES / f"{name}.{mode}.png"
                fig.savefig(out, facecolor=p.surface)
                paths.append(out)
            finally:
                plt.close(fig)
```

Every chart is drawn **twice**, once light and once dark, so the GitHub page
looks right whichever theme the reader uses.

`draw` is a function passed in. It receives the palette and must take all its
colours from it, which is what keeps the two versions consistent.

`finally: plt.close(fig)` always frees the memory, even if drawing crashes.
Without it, 28 charts leak memory.

The colours themselves come from a palette tested for colour-blindness, and the
comment records the measurements rather than claiming "these look nice".

---

# 10. The tests

**101 tests.** They all run on **invented data where the right answer is already
known**, so they need no internet and cannot break when a website changes format.

### Testing against known truth

```python
@pytest.mark.parametrize("true_r", [0.8, 1.0, 1.5, 2.5])
def test_rt_recovers_known_r(true_r):
    inc = _simulate_constant_r(true_r)
    est = estimate_rt(inc, tau=7)
    recovered = float(np.nanmean(est.mean[40:70]))
    assert recovered == pytest.approx(true_r, rel=0.02)
```

Build a fake epidemic where Rₜ is set to exactly 1.5, then check the code
recovers 1.5. If it returns 1.7, there is a bug.

`parametrize` runs the same test four times with four different values.

### Testing the hard case

```python
def test_ols_is_biased_under_nonlinear_confounding():
    """Establishes that the harder test below is actually hard."""
    df = _simulate(nonlinear=True, seed=2)
    est = ols_effect(df, "y", "t", ["x1", "x2"])
    assert abs(est.estimate - 0.5) > 0.05, "linear adjustment should not suffice here"


def test_dml_recovers_effect_under_nonlinear_confounding():
    df = _simulate(nonlinear=True, seed=2)
    est = dml_effect(df, "y", "t", ["x1", "x2"], n_folds=5, n_repeats=4)
    assert est.estimate == pytest.approx(0.5, abs=0.06)
```

This pair is my favourite.

The first test proves the simple method **fails** on this data. The second proves
the sophisticated method **succeeds**.

Without the first, the second would be unimpressive: maybe the problem was easy.
Together they show the hard test is genuinely hard.

### Testing the claim, not just the code

```python
def test_naive_zscore_flags_the_wave_peak_not_anomalies():
    """The central claim of Pillar 1, as an assertion.

    On a clean epidemic with no reporting artefacts at all, the textbook detector
    still fires -- and every flag is attributed to the trend.
    """
    s = _series(_multiwave())
    flags = naive_zscore_flags(s, threshold=3.0)
    assert flags.sum() > 0, "the naive detector fires on an artefact-free series"

    decomposed = decompose_flags(s, flags)
    assert (decomposed["mechanism"] == "trend").all()
    assert not decomposed["is_true_anomaly"].any()
```

Part 1's headline claim, written as a runnable assertion on a perfectly clean
synthetic epidemic. If someone ever breaks the decomposition, this test fails.

### Checking coverage empirically

```python
def test_conformal_intervals_attain_nominal_coverage_when_exchangeable():
    """Split conformal's guarantee holds under exchangeability; check it empirically."""
    ...
    covered = np.mean((np.array(truth) >= np.array(lower))
                      & (np.array(truth) <= np.array(upper)))
    assert covered == pytest.approx(0.90, abs=0.05)
```

Rather than trusting the theory, this generates 600 well-behaved cases and checks
the 90% intervals really do cover about 90%.

### The regression test

```python
def test_dml_accepts_a_custom_learner():
    """Passing an explicit learner must work.

    Regression test: the default was previously selected with ``learner or
    default``, and truth-testing a scikit-learn ensemble calls ``__len__``,
    which raises on an unfitted model. The bug was invisible while every caller
    passed None, and silently emptied the entire refutation suite the moment one
    did not.
    """
```

A real bug got caught here. Writing `learner or default()` looks fine, but
`or` asks Python "is this thing true?", and scikit-learn answers that by counting
trees in a forest that has not been built yet, which crashes.

The bug was invisible while every caller passed `None`. The moment one did not,
every robustness check silently returned nothing and the pipeline reported
"0 of 0 checks passed" as though that were a result.

Fixed to `learner is None`, plus this test so it can never come back, plus:

```python
    empty = [name for name, v in report.items()
             if isinstance(v, dict) and v.get("n") == 0]
    if empty:
        log.error("refutations produced no estimates: %s -- the estimator is "
                  "failing on every draw, not passing", ", ".join(empty))
```

Now an empty suite shouts instead of looking clean.

---

# 11. The runner scripts

Each script in `scripts/` is one button.

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entities", type=int, default=40)
    parser.add_argument("--quick", action="store_true",
                        help="fewer origins; for smoke-testing the pipeline")
    parser.add_argument("--rescore", action="store_true",
                        help="reuse the cached backtest and only redo scoring and figures")
    args = parser.parse_args()
```

`argparse` builds a command-line interface. `--rescore` exists because the
backtest takes 5 minutes but redrawing the charts takes 5 seconds. When tuning a
chart you do not want to redo the maths.

```python
    (TABLES / "forecast_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
```

Every stage writes its numbers to a file. That is what makes the rule "generate
every number, never type one in" real: each figure in the README traces back to a
file on disk.

---

# 12. How it all fits together

```
   fetch_data.py
        │  downloads 4 sources, saves fingerprints
        ▼
   run_forensics.py ──────────────► reliability_scorecard.csv
        │  scores 184 countries              │
        │                                    │
        ▼                                    │
   run_forecast.py ◄─────────────────────────┘
        │  30,384 forecasts, checks whether
        │  bad data means worse forecasts
        ▼
   run_causal.py
        │  4 estimators, 4 robustness checks,
        │  2 identification probes
        ▼
   run_capacity.py
           validates against real ICU data,
           ranks who needs beds
```

The arrow from forensics into forecasting is the part that makes this one project
rather than four. Part 1 produces a data-quality score. Part 2 uses it to show
that **countries with worse data are harder to forecast** (correlation −0.673).

Data quality is not a chore you finish before the real work. It caps how good the
real work can be.

## The one idea behind everything

> **A result is worth as much as the effort spent trying to prove it wrong.**

| Part | What it did to itself |
|---|---|
| 1 | Measured how badly the standard method fails, instead of assuming |
| 2 | Broke apart its own good result until the disappointing truth showed |
| 3 | Found a statistically perfect answer and demonstrated it points backwards |
| 4 | Showed exactly where its own model stops working, and why |

---

# Glossary

| Word | Plain meaning |
|---|---|
| **Backtest** | Pretend it is the past, make a prediction, then check against what really happened |
| **Baseline** | A deliberately simple method your clever method must beat |
| **Benford's law** | Naturally growing numbers start with 1 about 30% of the time, 9 only about 5% |
| **Bias** | A mistake that leans consistently one way, rather than cancelling out |
| **Caching** | Save the result so you never redo the slow work |
| **Calibration** | Whether "95% confident" is right 95% of the time |
| **Confounder** | A hidden third thing causing both things you are comparing |
| **Conformal prediction** | Build your error bars from your own past mistakes |
| **Convolution** | Look back over past days, weight each one, add them up |
| **Coverage** | How often reality actually lands inside your predicted range |
| **Cross-fitting** | Predict only data the model was not trained on |
| **DAG** | A diagram of arrows showing what causes what |
| **DML** | Double Machine Learning: strip out background factors, then look at what is left |
| **E-value** | How strong a hidden factor would need to be to explain your result away |
| **Identification** | Whether your setup can *in principle* answer the causal question |
| **IHR** | Infection-Hospitalisation Ratio: the share of infected people who go to hospital |
| **Kernel** | A list of weights saying how much each past day matters |
| **Log scale** | Counting by multiplying (1, 10, 100) instead of adding (1, 2, 3) |
| **Mediator** | Something *caused by* your treatment. Never adjust for one |
| **Median** | The middle value when sorted. Ignores extremes |
| **Monte Carlo** | Run a simulation many times with different guesses to see the range |
| **p-value** | If nothing were going on, how often would I see something this extreme by luck? |
| **Parquet** | A fast file format for tables |
| **Permutation test** | Shuffle the labels, see if the pattern survives. If it does, it was never real |
| **Placebo test** | Feed the method fake treatment and check it finds nothing |
| **Quantile** | A cut-point in a range. The 0.9 quantile is "90% of outcomes fall below this" |
| **Residual** | What is left over after your model explains what it can |
| **Reverse causality** | B causes A, but you assumed A causes B |
| **Rₜ** | How many people each infected person infects |
| **Seed** | The starting point that makes "random" repeatable |
| **Serial interval** | Days between one person catching a disease and passing it on |
| **SHA-256** | A fingerprint for a file. Change one character and it changes completely |
| **Synthetic control** | Build a fake comparison region from a blend of real ones |
| **WIS** | Weighted Interval Score: grades a forecast on accuracy *and* honest uncertainty |
| **z-score** | How many "typical spreads" above average a value sits |
