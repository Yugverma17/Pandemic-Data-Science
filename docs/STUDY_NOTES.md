# Study notes and interview preparation

Two halves.

**Part A** explains every concept this project uses, in the simplest words
possible. If you understand all of Part A, you understand the project.

**Part B** is interview questions with strong answers, sorted by role.

---

## Contents

**Part A: Every concept, explained simply**

- [A1. Numbers and averages](#a1-numbers-and-averages)
- [A2. Spread and distributions](#a2-spread-and-distributions)
- [A3. Probability and testing](#a3-probability-and-testing)
- [A4. Data quality tricks](#a4-data-quality-tricks)
- [A5. Time series](#a5-time-series)
- [A6. Forecasting](#a6-forecasting)
- [A7. Epidemiology](#a7-epidemiology)
- [A8. Causal inference](#a8-causal-inference)
- [A9. Machine learning](#a9-machine-learning)
- [A10. Software engineering](#a10-software-engineering)

**Part B: Interview questions**

- [B1. Opening questions](#b1-opening-questions-any-role)
- [B2. Data science and statistics](#b2-data-science-and-statistics)
- [B3. Machine learning](#b3-machine-learning)
- [B4. Causal inference](#b4-causal-inference)
- [B5. Software engineering](#b5-software-engineering)
- [B6. Hard follow-ups](#b6-hard-follow-ups)
- [B7. Questions to ask them](#b7-questions-to-ask-them)

---

# PART A: EVERY CONCEPT

---

## A1. Numbers and averages

### Mean (average)

Add everything up, divide by how many there are.

Five test scores: 10, 20, 30, 40, 100. Mean = 200 / 5 = **40**.

Notice something odd. Four of the five scores are *below* 40. One big number
dragged the average up.

### Median

Sort them, take the middle one.

10, 20, **30**, 40, 100. Median = **30**.

The median ignores extremes. That is exactly why this project uses it in several
places.

**Where it appears:** the backlog dump detector uses a rolling *median*, not a
mean. It is hunting for one giant spike, and a mean would let that spike drag the
"normal level" upward and hide itself. The median does not budge.

```python
med = x.rolling(window, center=True, min_periods=window // 2).median()
```

### When to use which

| Use | When |
|---|---|
| Mean | Data is well behaved, no crazy outliers |
| Median | Outliers exist and you do not want them steering the answer |

---

## A2. Spread and distributions

### Standard deviation (SD)

How spread out the numbers are.

- Class A scores: 49, 50, 51. Everyone is similar. **Small SD.**
- Class B scores: 10, 50, 90. Wildly different. **Big SD.**

Both classes have a mean of 50. The mean alone hides the difference.

### z-score

"How many standard deviations away from the mean is this?"

```
z = (value - mean) / standard deviation
```

If mean is 50 and SD is 10, then a score of 70 has z = 2. It is two spreads above
average.

**Rule of thumb:** about 99.7% of normal data sits within z = 3. So anything past
z = 3 looks unusual.

**Where it appears:** this is the popular anomaly detector this project tests and
finds nearly useless for COVID data.

```python
def naive_zscore_flags(new_cases: pd.Series, threshold: float = 3.0) -> pd.Series:
    mu, sd = x.mean(), x.std(ddof=0)
    return ((x - mu) / sd) > threshold
```

### Why the z-score fails here

> Imagine measuring someone's heart rate through a marathon. Near the finish
> their heart rate is far above their average for the day. A z-score flags those
> minutes as unusual. But nothing is wrong. They were running hard.

A z-score assumes the "normal level" stays put. In an epidemic it does not. It
moves by a factor of a thousand. So the z-score just finds the wave peaks.

### Distribution

A description of which values are common and which are rare.

- **Normal (bell curve):** most values near the middle, few at the edges. Heights
  of people.
- **Uniform:** every value equally likely. A fair dice roll.
- **Right-skewed:** most values small, a long tail of big ones. Salaries. Also
  hospital stays.

### Gamma distribution

A right-skewed shape used for waiting times. It cannot go below zero, which is
right for "days until something happens".

**Where it appears:** used three times.

| Used for | Mean |
|---|---|
| Serial interval (days between infections) | 4.7 |
| Case to hospital admission delay | 7 |
| ICU length of stay | 12 |

### Percentile and quantile

The same idea, different scale.

"90th percentile" = "0.9 quantile" = 90% of values sit below this point.

If you are at the 90th percentile of height, you are taller than 90 out of 100
people.

**Where it appears:** every forecast is reported as seven quantiles.

```python
QUANTILES = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
```

The 0.50 quantile is the median, the middle guess. The gap between 0.025 and
0.975 is a 95% range.

### Log scale

Counting by multiplying instead of adding.

- Normal scale: 1, 2, 3, 4, 5
- Log scale: 1, 10, 100, 1000

**Why it matters here.** Epidemics multiply. Cases go 100, 200, 400, 800. On a
log scale that becomes a straight line, which is far easier to model.

Logs turn multiplying into adding, which is the whole trick.

```python
logx = np.log1p(x.clip(lower=0))
```

`log1p(x)` means "log of (x + 1)". The +1 exists because log(0) is undefined and
case counts do hit zero.

### Geometric mean

The right average when you are averaging *ratios*.

A model is twice as good on one country and twice as bad on another. It should
come out even.

| Method | Calculation | Result | Verdict |
|---|---|---|---|
| Ordinary mean | (0.5 + 2.0) / 2 | 1.25 | Says "worse". Wrong |
| Geometric mean | sqrt(0.5 x 2.0) | 1.00 | Correct |

```python
return float(np.exp(np.mean(np.log((s[ok] + eps) / (b[ok] + eps)))))
```

Take logs, average, undo the log. That is the standard way to compute it.

---

## A3. Probability and testing

### p-value

The single most misunderstood number in statistics. Here is what it actually
means:

> **If nothing unusual were going on, how often would I see something this
> extreme purely by luck?**

- p = 0.5 means "half the time". Totally normal.
- p = 0.05 means "1 time in 20". Slightly surprising.
- p = 0.001 means "1 time in 1000". Very surprising.

**What p does NOT mean:** it is not the probability your idea is right. It only
says how surprising your data is if nothing were happening.

**Where it appears:** Egypt's last digits give p = 6.8e-22. That is
0.00000000000000000000068. If the digits were genuinely random, you would
essentially never see a pattern that lopsided.

### Chi-square test

A way to check "does what I observed match what I expected?"

For each category: take the gap, square it, divide by what you expected. Add
them all up.

```python
chi2 = float(((observed - expected) ** 2 / expected).sum())
```

- **Squaring** means being too high and too low both count as bad.
- **Dividing by expected** is the clever part. A gap of 10 is huge if you expected
  20, and trivial if you expected 2000.

### Confidence interval

A range instead of a single number.

"The effect is 0.035" is overconfident. "The effect is between 0.018 and 0.051"
is honest.

### Permutation test

A beautiful idea that needs almost no maths.

**The problem:** you see a weekly pattern in the data. Is it real, or could it
happen by chance?

**The solution:** destroy the pattern on purpose and see how often chance
recreates it.

1. Shuffle the day labels randomly. Monday's numbers become Thursday's.
2. Any weekly pattern is now destroyed by definition.
3. Measure the pattern strength anyway.
4. Repeat 500 times.
5. Ask: how often did random shuffling beat the real thing?

```python
    for i in range(n_permutations):
        null[i] = amplitude_of(rng.permutation(r))[1]
    p_value = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
```

**Why the +1 on top and bottom?** Without it you could report p = 0, which claims
"impossible by chance". With only 500 shuffles you cannot know that. The +1 caps
the strongest claim you can make at about 0.002.

**Why use this instead of a standard test?** Standard tests assume your data has a
nice bell-curve shape. This data does not. A permutation test assumes nothing. It
just shuffles.

### Correlation

Do two things move together?

- **+1**: perfectly together. More study hours, higher marks.
- **0**: unrelated.
- **-1**: perfectly opposite. More time gaming, lower marks.

### Pearson vs Spearman

| Type | Measures | Use when |
|---|---|---|
| Pearson | Straight-line relationship | The link is roughly linear |
| Spearman | Rank relationship | You only care about order, or the link is curved |

**Where it appears:** Spearman is used to check whether the reliability ranking
changes when the weights change, because only the *order* matters there.

### Dirichlet distribution

A way to generate random sets of numbers that always add up to 1.

**Where it appears:** the reliability index combines seven detectors with seven
weights that sum to 1. Those weights are a judgement call. So the code draws 500
random weight sets from a Dirichlet centred on the chosen ones, rebuilds the
ranking each time, and checks the ranking barely moves.

Result: **median Spearman 0.958.** The ordering comes from the data, not from the
weights.

---

## A4. Data quality tricks

### Benford's law

A genuinely strange fact.

In numbers that grow naturally, the **first digit** is not evenly spread:

| First digit | How often |
|---|---|
| 1 | about 30% |
| 2 | about 18% |
| ... | ... |
| 9 | about 5% |

**Why?** Think about a number growing from 100. It must pass through 100 to 199,
all starting with 1, before it reaches 200. That is a wide stretch. But 900 to
1000 goes by quickly. So numbers spend more time starting with 1.

```python
BENFORD_P = np.log10(1.0 + 1.0 / np.arange(1, 10))
```

Numbers people invent do not do this. Made-up digits come out too even.

**Important caveat this project states openly:** failing Benford is a reason to
look closer, never proof of cheating. A short series stuck in one range fails it
innocently. That is why Benford gets the **smallest weight** of the seven
detectors, 0.05.

### Terminal digit test

Stronger than Benford, and simpler.

Take a big count like 4,873. The **last** digit carries no information at all.
Across thousands of days each digit 0 to 9 should appear about 10% of the time.

If it does not, someone rounded, estimated, or typed the number rather than
counting it.

```python
last = np.mod(v.astype(np.int64), 10)
```

`np.mod(x, 10)` gives the remainder after dividing by 10, which is the last digit.

**Why is this stronger than Benford?** Benford needs the numbers to span several
orders of magnitude. The last-digit test only needs them to be big. Fewer
assumptions means stronger evidence.

### Digit heaping

When people estimate, they round to 0 or 5. "About 50 cases." "Roughly 200."

So an excess of numbers ending in 0 or 5 is a fingerprint of estimation.

---

## A5. Time series

### What it is

Data with a time order, where the order matters. Daily cases. Stock prices. Your
weight each morning.

You cannot shuffle a time series. Tuesday must come after Monday.

### Trend, seasonality, residual

Any time series splits into three parts:

```
today = general level  +  repeating pattern  +  what is left over
```

- **Trend**: the slow direction. Cases rising over a month.
- **Seasonality**: a repeating cycle. Fewer cases reported on Sundays.
- **Residual**: everything the first two cannot explain.

**This decomposition is the heart of Part 1.**

```python
    trend = logx.rolling(7, center=True, min_periods=7).mean()
    dev = logx - trend
    wk_effect = dev.groupby(dow).transform("mean")
    resid = dev - wk_effect
```

Only the residual can be a genuine surprise. The z-score detector cannot tell
these apart, which is why 111 of its 112 flags turn out to be trend or weekday.

### Rolling window

Look at a moving chunk of days instead of all of them.

A 7-day rolling mean on 1, 2, 3, 4, 5, 6, 7, 8 gives:
- Days 1 to 7: mean = 4
- Days 2 to 8: mean = 5

### Centred vs trailing

| Type | Uses | Good for |
|---|---|---|
| **Trailing** | Only past days | Forecasting. You cannot see the future |
| **Centred** | Days before and after | Analysing history. Smoother, no lag |

**This distinction is critical.** Using a centred window inside a forecast would
be cheating, because it peeks at days that have not happened yet.

The project uses centred windows for *analysing* the past (Part 1) and trailing
windows for *forecasting* (Part 2). Getting these backwards is a classic bug.

### Why exactly 7 days

Because a week has 7 days. Averaging over exactly one full week makes the weekly
pattern cancel out perfectly. A 6-day or 8-day window would leave some of it
behind.

### Lag

How far back you look. "The 3-day lag" means the value from 3 days ago.

### Autocorrelation

Whether a series is correlated with its own past. Today's cases look a lot like
yesterday's, so COVID data has very high autocorrelation.

This is why "next week looks like this week" is such a hard baseline to beat.

---

## A6. Forecasting

### Baseline

A deliberately simple method your clever method must beat.

```python
class Persistence(Forecaster):
    """"Tomorrow looks like today." The level stays where it is."""
    def predict(self, cache: SeriesCache, i: int, horizon: int) -> float:
        return cache.level(i)
```

The entire model is one line.

**Why does this matter so much?** Because without a baseline, "my model has 85%
accuracy" is meaningless. If guessing gets 84%, your model is worthless. Many
published models never check.

### Horizon

How far ahead you predict. This project uses 7 and 14 days.

Longer horizon means harder. Always report the horizon, because "90% accurate" is
meaningless without it.

### Backtest

Pretend it is the past, make a prediction, then check against what really
happened.

### Rolling origin

Do that many times, moving the pretend "today" forward each time.

```
Origin 1: use data up to Jan 1  -> predict Jan 15  -> compare
Origin 2: use data up to Jan 15 -> predict Jan 29  -> compare
Origin 3: use data up to Jan 29 -> predict Feb 12  -> compare
```

This project uses 74 origins across 40 countries, giving 30,384 forecasts.

### Lookahead bias (data leakage)

Accidentally using information you would not have had at the time.

**This is the number one way forecasting projects fool themselves.** The model
looks brilliant in testing and collapses in real use.

Three places it can sneak in, all of which this project blocks:

**1. In the features.** Using a centred average that peeks forward. Blocked by
using only causal quantities, and there is a test that proves it:

```python
def test_rt_estimate_is_causal():
    """R_t at index i must not change when future data is appended or altered."""
    perturbed = inc.copy()
    perturbed[200:] *= 7.0
    np.testing.assert_allclose(estimate_rt(perturbed, tau=7).mean[:200],
                               full[:200], rtol=1e-9, atol=1e-9)
```

It multiplies all the future data by 7 and checks the past does not move.

**2. In the training data.** Training on rows whose *target* is in the future.
Blocked by only training on rows whose target date is before the cutoff.

**3. In the error bars.** The subtlest one. If a 14-day forecast is made on the
1st, you do not learn whether it was right until the 15th. So that error cannot
be used to tune a forecast made on the 5th.

```python
            for item in pending:
                if item[0] <= origin:
                    calibrator.add_residual(mname, entity, horizon, actual, point)
                else:
                    still_pending.append(item)
```

Errors sit in a waiting queue and are released only when the clock reaches their
target date.

### Prediction interval

The range around a forecast. "Probably 5,000, likely between 3,000 and 8,000."

### Coverage and calibration

**Coverage:** how often reality actually lands inside your range.

**Calibration:** whether your stated confidence is honest. A calibrated 95%
interval contains the truth 95% of the time.

**The project's honest result:**

| Stated | Actual |
|---|---|
| 95% | about 85% |
| 50% | 43 to 45% |

Every model is overconfident, and the README says so instead of hiding it.

### Conformal prediction

Build your error bars from your **own past mistakes** rather than from a
theoretical formula.

"I usually land within 30% either way, so my range is my guess plus or minus 30%."

```python
    offsets = _empirical_quantiles(residuals, levels)
    return np.clip(np.expm1(np.log1p(max(point, 0.0)) + offsets), 0.0, None)
```

**Why it is good:** its guarantee does not depend on errors being bell-shaped,
which these are not.

**Its one assumption:** exchangeability, meaning past errors resemble future
ones. Epidemics break this when a new variant arrives. The project measures how
badly rather than assuming it away.

### WIS (Weighted Interval Score)

The grade for a forecast. Lower is better.

Plain accuracy cannot tell apart:
- a model that is right and honest about uncertainty
- a model that is right but occasionally wildly overconfident

The second is dangerous. WIS notices.

```python
def interval_score(y, lower, upper, alpha):
    width = upper - lower
    under = (2.0 / alpha) * np.clip(lower - y, 0, None)
    over = (2.0 / alpha) * np.clip(y - upper, 0, None)
    return width + under + over
```

Three parts:

1. **width**: a penalty just for being vague. "Between 0 and a million" is always
   right and completely useless.
2. **under**: punishment if reality came in below your range.
3. **over**: punishment if reality came in above.

The `2/alpha` factor is the clever design:

| Interval | alpha | Penalty multiplier |
|---|---|---|
| 95% | 0.05 | **40** |
| 50% | 0.50 | **4** |

**Being wrong when you claimed high confidence hurts ten times more.** Exactly
right.

### Relative skill

Your score divided by the baseline's score, on the same task.

- 0.77 means 23% better than the baseline
- 1.04 means **worse** than doing nothing

Comparing on the same task matters because India has thousands of times more
cases than Iceland. Raw error would just rank countries by size.

---

## A7. Epidemiology

### Rt (the reproduction number)

How many people each infected person passes it to.

| Rt | Meaning |
|---|---|
| 2.0 | Each person infects 2. Cases double |
| 1.0 | Each infects 1. Flat |
| 0.8 | Shrinking |

**Rt = 1 is the tipping point.** Everything above grows, everything below shrinks.

### Serial interval

Days between one person catching a disease and passing it on. About 4.7 days for
COVID.

### Renewal equation

The maths connecting them:

```
today's cases = Rt x (how infectious the recent past is)
```

Where "how infectious the recent past is" weights each past day by how likely it
is to cause a case today.

```python
def total_infectiousness(incidence: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Lambda_t = sum_{s>=1} I_{t-s} w_s, by direct convolution."""
```

### Why w[0] must be zero

```python
    w[0] = 0.0
```

You cannot infect someone on the same day you were infected. Without this line,
today's cases would feed into their own calculation and Rt would be dragged
artificially toward 1.

One line, and the whole estimate depends on it.

### Discretisation bias

A trap this project explicitly avoids.

We know the gap between infections averages 4.7 days. Our data is daily, so we
need a list: what fraction happen exactly 1 day later, 2 days, 3 days?

The **obvious** conversion is wrong:

> binning a Gamma CDF at integer edges (`w_s = F(s) - F(s-1)`) yields a
> distribution whose mean is half a day too large, because it assigns the mass of
> `[s-1, s)` to the point `s`

It quietly adds **half a day** to every gap, and that flows into every Rt
estimate. The project uses the published correction and has a test proving the
mean comes out at exactly 4.7.

### IHR (Infection Hospitalisation Ratio)

The share of infected people who end up in hospital. About 3%, somewhere between
1.5% and 6%.

### Length of stay

How long a patient stays. ICU stays average about 12 days and are heavily
right-skewed, meaning most are shorter but some are very long.

### Convolution

Sounds terrifying, is simple:

> For today's total, look back at every past day, weight it by how much it
> contributes, and add it all up.

```python
def _convolve_causal(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution: ``y[t] = sum_s x[t-s] * kernel[s]``, same length as x."""
    full = np.convolve(x, np.asarray(kernel, float), mode="full")
    return full[: x.size]
```

`full[: x.size]` chops off the tail so today can never depend on tomorrow.

### The car park idea

**The most important intuition in Part 4.**

> Cars arriving per hour is one thing. Cars *currently parked* is another. If
> everyone stays 12 hours, the car park keeps filling long after arrivals slow.

ICU occupancy is not the same as ICU admissions. Because stays average 12 days,
**occupancy keeps rising for one to two weeks after cases have peaked.**

A model treating ICU as "cases times a number" calls the turning point a
fortnight early. Which is exactly when you need to decide about surge capacity.

Measured across 202 countries, the gap between the case peak and the ICU peak is
**13 days**. That is the warning the model buys.

### Survival function

The chance something has *not* ended yet.

`S[s]` = chance a patient admitted `s` days ago is still there.

- `S[0]` = 1.0, everyone admitted today is still here
- `S[12]` = about 0.5
- `S[40]` = small

The area under this curve is the mean stay.

---

## A8. Causal inference

**The hardest and most valuable section.**

### Correlation is not causation

Ice cream sales and drowning deaths rise together. Ice cream does not cause
drowning. **Summer** causes both.

### Confounder

The hidden third thing causing both. Summer, in that example.

### DAG (Directed Acyclic Graph)

A diagram of arrows showing what causes what.

- **Directed**: arrows point one way
- **Acyclic**: no loops, nothing causes itself
- **Graph**: dots and arrows

```python
EDGES: list[tuple[str, str]] = [
    ("median_age", "deaths"),
    ("median_age", "stringency"),
    ("wealth", "stringency"),
    ...
]
```

**Why write it down?** A list of control variables hides your assumptions. A
picture with arrows makes every assumption visible, so someone who disagrees can
point at exactly which arrow is wrong.

### Mediator

Something **caused by** your treatment, sitting between it and the outcome.

```
lockdown  ->  fewer people meeting  ->  fewer deaths
                    (mediator)
```

**Never control for a mediator.**

> It is like asking "did exercise improve health, ignoring any change in
> fitness?" You just deleted the effect you were measuring.

The project's DAG deliberately includes testing, transmission, observed cases and
vaccination speed **so the rule can refuse them**:

```python
    descendants = nx.descendants(g, treatment)
    bad = adjustment & descendants
    if bad:
        return False, (f"post-treatment variables in the adjustment set: {sorted(bad)} "
                       "-- these are mediators, ...")
```

### Collider

The opposite trap. Something **caused by two things**. Controlling for a collider
*creates* a fake link that was not there.

Rule of thumb: adjust for common **causes**, never for common **effects**.

### Back-door criterion

Pearl's rule for choosing what to control for:

> Z is admissible if
>   (i) no member of Z is a descendant of T, and
>   (ii) Z blocks every back-door path from T to Y

In plain words: control for the confounders, never for the consequences.

The project implements this in code and unit tests it, rather than just claiming
an answer.

### Reverse causality and simultaneity

You assumed A causes B, but really B causes A, or something decides both at once.

**This is the project's central finding.**

> It is like noticing that ambulances are found wherever accidents happen, and
> concluding that ambulances cause accidents.

Governments locked down **because** cases were exploding. So countries with
strict lockdowns are, by definition, the ones already in trouble.

Measured directly:

```python
    concurrent = ols_effect(
        sub, outcome=treatment, treatment=concurrent_col, ...
        method="stringency ~ outbreak size during the window")
```

Result: **+2.26, p = 0.032.**

### ATE (Average Treatment Effect)

The average difference the treatment makes across everyone.

### OLS (Ordinary Least Squares)

Plain linear regression. Fit the best straight line.

**Its weakness:** it assumes everything works in straight lines. COVID risk
against age is not a straight line, it rises steeply. The leftover mess
contaminates the answer.

### DML (Double Machine Learning)

Lets flexible models handle the background factors while still giving a
trustworthy answer for the one thing you care about.

**Three steps:**

1. Predict **deaths** from background factors. Keep the leftover.
2. Predict **lockdown** from the same factors. Keep the leftover.
3. See whether the two leftovers move together.

```python
        y_res = y - y_hat
        t_res = t - t_hat
        denom = float(np.mean(t_res**2))
        theta = float(np.mean(t_res * y_res) / denom)
```

> Take out everything the background explains. Whatever still lines up afterwards
> is the bit we actually care about.

**Why "double"?** Because you model two things, not one: the outcome *and* the
treatment.

### Cross-fitting

Predict only data the model was **not** trained on.

```python
        folds = KFold(n_splits=n_folds, shuffle=True, random_state=rng_seed)
```

Split into 5 groups. Predictions for group 1 come from a model trained on groups
2 to 5.

**Why?** A flexible model partly memorises its training data. If it predicted the
same rows it learned from, its leftovers would be artificially small and the
final answer biased.

### Propensity score and IPW

**Propensity score:** the chance a unit got the treatment, given its
characteristics.

**IPW (Inverse Probability Weighting):** give more weight to surprising cases. A
poor country that locked down hard is informative, so it counts for more.

### Placebo test

Feed the method fake treatment and check it finds nothing.

```python
        d[treatment] = r.permutation(d[treatment].to_numpy())
```

Shuffle lockdown values randomly between countries. **If a real effect exists, it
must vanish.**

Result: shuffled average +0.0009 versus real +0.0347. It vanishes. Pass.

### E-value

How strong a hidden factor would need to be to explain your result away entirely.

```python
    def _ev(rr: float) -> float:
        if rr < 1:
            rr = 1.0 / rr
        return float(rr + np.sqrt(rr * (rr - 1.0)))
```

- Near 1: a weak hidden factor is enough. Fragile.
- 4 or more: the hidden factor would have to beat everything you measured.

Here: **2.18.** Moderate.

### Robustness vs identification

**The single most important idea in this project.**

| Concept | Question it answers |
|---|---|
| **Robustness** | Is the number *stable* when I poke it? |
| **Identification** | Can this study design answer the causal question *at all*? |

> an estimate driven entirely by reverse causality is perfectly stable, survives
> every placebo, and has a large E-value. Stability and identification are
> different properties, and conflating them is how a confounded number acquires a
> confident standard error.

In this project **all four robustness checks pass** and the estimate is **still
not causal**. Knowing the difference is what separates a careful analyst from
someone who reports p < 0.001 and stops.

### Synthetic control

When one region does something and no single other region is a fair comparison,
**build** one from a blend of others.

```python
    result = minimize(
        loss, w0, jac=grad, method="SLSQP",
        bounds=[(0.0, 1.0)] * n_donors,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, ...}],
    )
```

Two rules do the work:

- **No negative weights.** You cannot use "minus 30% of Kerala".
- **Weights sum to 1.**

Together they mean the fake region can only be a weighted average of things that
actually happened. It can never invent a scenario outside the real data.

### Parallel trends

The assumption behind these methods: without the intervention, treated and
comparison would have moved together.

**You can check this before the intervention.** If they were already diverging,
the design is broken.

That is exactly what happened here. Maharashtra's synthetic twin stopped matching
it three weeks *before* the lockdown, so the post-period gap measures
pre-existing divergence, not policy.

---

## A9. Machine learning

### Supervised learning

Learn from examples where you know the answer, then predict new cases.

### Features and target

- **Features**: the inputs. Growth rate, current level, Rt.
- **Target**: what you predict. Cases in 14 days.

### Feature engineering

Building better inputs from raw data. Usually matters more than the algorithm.

**The key decision in this project's ML model** is not which algorithm, it is
*what to predict*:

> Predicts the *log growth* from origin to target rather than the level, which
> matters more than the choice of learner: predicting a level forces the model to
> spend capacity re-learning each country's scale, whereas predicting a ratio lets
> every country's wave contribute to one shared question.

Predicting the **level** wastes the model's effort learning that India is big and
Iceland is small. Predicting the **growth ratio** lets every country's wave answer
the same question.

### Overfitting

Memorising the training data instead of learning the pattern.

Like memorising past exam answers rather than understanding the subject. Perfect
on the practice paper, lost on the real one.

**Signs:** great on training data, poor on new data.

### Regularisation

Deliberately limiting the model so it cannot memorise.

```python
        self._model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=self.seed,
        )
```

- `max_depth=6`: trees cannot get too complicated
- `min_samples_leaf=40`: every decision must be based on at least 40 examples
- `early_stopping`: stop when it stops improving on held-out data

### Decision tree, random forest, gradient boosting

- **Decision tree**: a flowchart of yes/no questions.
- **Random forest**: many trees, each on a random slice of the data, then average.
  Reduces overfitting.
- **Gradient boosting**: trees built one after another, each fixing the previous
  one's mistakes. Usually more accurate, more prone to overfitting.

### Bias-variance tradeoff

- **Bias**: too simple, misses real patterns. Underfitting.
- **Variance**: too complex, chases noise. Overfitting.

You cannot minimise both. The art is balancing them.

### Why trees cannot extrapolate

A tree predicts by averaging the training examples in each leaf. It can never
output a value outside the range it has seen.

That is why the project clips its output:

```python
        log_growth = float(np.clip(self._model.predict(x)[0], -2.0, 2.0))
```

### Train/validation/test

- **Train**: the model learns from this
- **Validation**: tune settings on this
- **Test**: touch exactly once, at the end

**For time series, these must be split by time, never randomly.** A random split
lets the model learn from the future.

---

## A10. Software engineering

### Version control (git)

Tracks every change, lets you go back, lets people work together.

- **commit**: a saved snapshot with a message
- **branch**: a parallel line of work
- **push / pull**: send to / get from a shared server

### Unit test

A small check that one piece of code is correct.

The strongest kind used here: **test against known truth.**

```python
@pytest.mark.parametrize("true_r", [0.8, 1.0, 1.5, 2.5])
def test_rt_recovers_known_r(true_r):
    inc = _simulate_constant_r(true_r)
    est = estimate_rt(inc, tau=7)
    recovered = float(np.nanmean(est.mean[40:70]))
    assert recovered == pytest.approx(true_r, rel=0.02)
```

Build a fake epidemic where Rt is exactly 1.5, then check the code recovers 1.5.

### Regression test

A test written *after* fixing a bug, to make sure it never returns.

```python
def test_dml_accepts_a_custom_learner():
    """Regression test: the default was previously selected with ``learner or
    default``, and truth-testing a scikit-learn ensemble calls ``__len__``,
    which raises on an unfitted model."""
```

### Test doubles and synthetic data

All 101 tests here run on **invented data with known answers**. So they:
- need no internet
- run in under a minute
- cannot break when a website changes format

That last point matters. A test suite that quietly stops testing anything is
worse than no tests.

### CI (Continuous Integration)

A robot that runs your tests automatically on every push.

This project runs lint plus 101 tests on Python 3.11, 3.12 and 3.13, plus a check
that every module imports.

### Caching

Save the result so you never redo slow work.

```python
    if src.path.exists() and not force:
        log.info("cache hit  %-18s %s", key, _human(src.path.stat().st_size))
        return src.path
```

First run downloads 150 MB. Every run after takes zero seconds.

### Atomic writes

Write to a temporary file, then rename only when finished.

```python
    tmp = src.path.with_suffix(src.path.suffix + ".part")
    ...
    tmp.replace(src.path)
```

**Why:** if your wifi dies at 80%, without this you get a half-file with the
correct name. Next run sees the name, assumes it is complete, and silently
analyses broken data.

### Idempotency

Running something twice gives the same result as running it once. Safe to retry.

### Reproducibility

Someone else can get your exact numbers.

Needs: fixed seeds, pinned library versions, recorded data.

### SHA-256 hash

A fingerprint for a file. Change one character and the fingerprint changes
completely.

```python
        "sha256": _sha256(src.path),
```

Lets you prove months later exactly which data produced a chart.

### Dependency pinning

Recording exact library versions so the code still works next year.

```
numpy~=2.1          # tested 2.1.3
pandas~=3.0         # tested 3.0.5
```

`~=2.1` means "2.1 or newer, but not 3.0". Patches fine, major versions not.

### Linting

Automatic style and error checking. This project uses `ruff`.

### Type hints

Saying what type each thing is.

```python
def peak_lag_days(cases: np.ndarray, census: np.ndarray) -> int:
```

Takes two arrays, returns an integer. Catches mistakes early and documents itself.

### Dataclass

A shortcut for classes that mainly hold data.

```python
@dataclass(frozen=True)
class EpiParams:
    serial_interval_mean: float = 4.7
```

`frozen=True` means the values **cannot be changed while running**. Constants
should stay constant.

### Context manager

The `with` block. Guarantees cleanup even if something crashes.

```python
@contextlib.contextmanager
def use_theme(mode: str):
    with mpl.rc_context(rc):
        yield p
```

### Separation of concerns

Each piece does one job.

- `src/pandemic/` is the **recipe book**, it knows how to do things
- `scripts/` are the **meals**, they combine recipes into a result

This is why 101 tests can run in under a minute without touching real data.

### Fail loudly

If something breaks, shout. Do not silently return nothing.

A real bug in this project made every robustness check return nothing while the
pipeline printed "0 of 0 checks passed" as if that were a result. The fix:

```python
    if empty:
        log.error("refutations produced no estimates: %s -- the estimator is "
                  "failing on every draw, not passing", ", ".join(empty))
```

---

# PART B: INTERVIEW QUESTIONS

How to use this section: read the question, try to answer it yourself first,
then check against the answer given. The answers are written the way you should
actually say them out loud, not as an essay. Where a number is quoted, it is a
real number this project produced, not a placeholder.

---

## B1. Opening questions (any role)

### "Walk me through this project."

> I built a pipeline that answers four questions about COVID data: can we catch
> reporting errors, can we forecast case growth, can we find why some regions did
> better, and can we predict hospital strain. The thing that makes it different
> from a typical project is that for every claim I also ran the test that could
> have proven it wrong, and I kept the results even when they were disappointing.
> For example, the standard anomaly detector everyone uses turns out to have
> about 1% precision on real reporting failures. And my causal estimate for
> lockdowns survives four separate robustness checks and is still not causal,
> because I can show the policy was decided by the same thing that decided the
> outcome. That second finding is the one I'm proudest of, because it required
> knowing that "robust" and "correct" are different properties.

### "Why did you build this?"

> I wanted a portfolio project that would hold up under real questioning, not
> just one that produces charts. Most COVID dashboards trust the numbers. I
> wanted to build one that questions them, and one where every conclusion is
> something I can defend line by line if someone pushes back.

### "What was the hardest part?"

> Making the forecast backtest honest. It's very easy to accidentally let a
> model see information it wouldn't have had at the time. I had to build a
> queue that holds every prediction error until its actual target date arrives,
> because if you calibrate a 14-day forecast using an error you could only have
> known about after 14 days, you're training with the answer key. I also wrote a
> test that proves my reproduction-number calculation is causal, by feeding it
> altered future data and checking the past output doesn't move by even one part
> in a billion. Without that guarantee the whole backtest would have been
> meaningless.

### "What would you do differently if you had more time?"

> Two things. First, the causal analysis is stuck at the country level, and
> country-level policy timing is too tangled up with country-level severity to
> ever separate cleanly. I'd want district-level data with genuinely staggered
> policy rollout dates, which would support a proper difference-in-differences
> design. Second, the forecast models only really add value when cases are
> falling. I'd want to try a model that explicitly detects regime changes,
> rather than one that averages over all of them.

---

## B2. Data science and statistics

### "How would you detect anomalies in a time series?"

> It depends what you mean by anomaly, and that's actually the trap. A plain
> z-score flags points far from the mean, but in a series with a strong trend,
> like an epidemic wave, that just finds the peak of the wave again. I proved
> this directly: on a clean simulated epidemic with zero data errors, a 3-sigma
> z-score still fires, because the wave peak is genuinely far from the average.
> The fix is to decompose the series first, trend plus seasonality plus
> residual, and only treat the residual as a candidate anomaly. On real data
> that cut a 112-day flag list down to 1 real anomaly, while separately catching
> 43 real reporting failures the z-score missed completely.

### "What's the difference between correlation and causation, with an example from your project?"

> Correlation just means two things move together. Causation means one
> actually makes the other happen. My project has a clean example: countries
> that responded to COVID more strictly also had more deaths, correlation of
> about +0.035 per stringency point, and that holds up under four different
> robustness checks. But it's not causal, because governments impose stricter
> measures specifically *because* an outbreak is already severe. I proved this
> by regressing lockdown strictness on the size of the outbreak happening at the
> same time, and got a strong positive relationship, p equals 0.032. So the
> policy and the outcome are both driven by the same thing, the severity of the
> outbreak, and treating the correlation as causal would get the direction of
> the story backwards.

### "Explain a p-value to a non-technical person."

> It's the answer to one specific question: if nothing unusual were actually
> happening, how often would I see data this extreme just by chance? A small
> p-value means "very rarely by chance," so something real is probably going on.
> It's not the probability that your theory is correct, that's a common mix-up.
> In my project, one country's last digits give a p-value of about 10 to the
> minus 22. That means if the digits were genuinely random, you'd essentially
> never see a pattern that lopsided by luck.

### "Why use the median instead of the mean in some of your detectors?"

> Because I'm specifically hunting for one huge spike, and the mean is exactly
> the statistic a spike distorts. If I used a rolling mean to define "the normal
> level" and then looked for days far above it, the spike itself would drag the
> normal level upward and partly hide itself. The median ignores extremes, so
> the "normal level" stays stable even with a spike sitting right next to it.

### "What is a permutation test, and why not just use a standard statistical test?"

> A permutation test checks whether a pattern is real by destroying it on
> purpose and seeing how often chance recreates it. I shuffle the day-of-week
> labels, which destroys any real weekly pattern, measure the pattern strength
> anyway, and repeat that 500 times. If the real pattern is stronger than
> essentially all of the shuffled ones, it's real. I used this instead of a
> standard test because those assume your data has a roughly normal shape, and
> daily case ratios are heavily skewed. A permutation test makes no assumption
> about the shape of the data at all, it just shuffles.

### "How do you handle imbalanced or weighted judgment calls in a scoring system?"

> In my reliability index I combine seven detectors with seven weights I chose
> myself, and any hand-picked weight is a place reviewers will push back. So I
> stress-tested my own judgment: I redrew the weight vector 500 times from a
> random distribution centred on my choices, rebuilt the country ranking each
> time, and measured how much the ranking moved. The median rank correlation
> against my original ranking was 0.958, so the ordering is coming from the
> data, not from my specific weight choices. That's the difference between
> asking someone to trust your weights and giving them evidence they don't need
> to.

### "What's the difference between precision and recall, and can you give an example where you calculated both?"

> Precision is: of the things you flagged, how many were actually right.
> Recall is: of the things that were actually true, how many did you catch.
> In my project, the standard z-score anomaly detector flags 112 days across
> four countries. Only 1 of those turns out to be a genuine surprise once you
> account for trend and weekday effects, so precision is roughly 1%. Separately,
> I know there are 43 real reporting events in those same countries from my own
> detectors, and the z-score caught zero of them, so recall is 0%. That's a
> method that is bad in both directions at once, and reporting both numbers
> makes that unambiguous in a way that reporting just one would not.

---

## B3. Machine learning

### "Why did your gradient boosting model not beat the simpler epidemiological model?"

> Because it didn't have enough data relative to the task. It was trained
> across all 40 countries, but the mechanistic model gets to use a real
> epidemiological structure, the renewal equation, that doesn't need to be
> learned from scratch, it's already known to be approximately true. The
> gradient boosting model has to learn everything from patterns in the data,
> and it came close, 0.792 versus 0.769 relative WIS at 7 days, but it never
> quite caught up. The lesson I took from that is that domain knowledge is a
> form of extremely efficient training data, and when you have real domain
> knowledge available, it's worth trying before reaching for a general learner.

### "How did you prevent your forecasting backtest from leaking future information?"

> Three separate places, and I have a specific defence for each. First, every
> feature the models use is provably causal, meaning a value at day i depends
> only on days before i, and I have a test that proves this by feeding the model
> altered future data and checking the past output is bit-identical. Second, the
> machine learning model is only ever trained on rows whose *target* date is
> before the cutoff, not just whose feature date is before the cutoff, which is
> the more common and subtler mistake. Third, and this is the one people miss
> most often, the prediction intervals are built from the model's own past
> errors, but an error from a 14-day-ahead forecast isn't observable until 14
> days later. So I hold every error in a pending queue and only release it into
> the calibration pool once the clock reaches its actual target date.

### "What is regularisation and how did you use it?"

> Regularisation is deliberately limiting a model so it can't just memorise the
> training data. I used several forms on the gradient boosting model: capped
> tree depth at 6, required at least 40 examples per leaf so no decision rests
> on a handful of points, added an L2 penalty, and used early stopping against a
> held-out validation slice. Given that model only had 40 countries worth of
> data, without that it would have overfit badly.

### "Your model predicts log growth instead of the level. Why does that matter?"

> Because predicting the level forces the model to spend most of its capacity
> just learning that India is a big country and Iceland is a small one, which
> isn't useful information, it's just re-deriving population size. Predicting
> the growth ratio instead means every country's wave, big or small, is
> answering the exact same underlying question: given this growth pattern, what
> happens next? That lets 40 countries' worth of waves all contribute to
> learning one shared relationship, instead of 40 countries each contributing a
> little bit of noisy evidence about their own scale.

### "How would you evaluate whether a forecasting model is actually useful, beyond just accuracy?"

> Accuracy on its own hides overconfidence. I used the Weighted Interval Score,
> which is the metric the official COVID-19 Forecast Hubs used, because it
> scores both how close your point estimate was and how honest your uncertainty
> range was. A model that's usually right but occasionally catastrophically
> overconfident should score worse than a model that's slightly less accurate
> but knows its own limits, and plain accuracy can't tell those apart. I also
> checked calibration directly: does the 95% interval actually contain the
> truth 95% of the time? In my project it was closer to 85%, and I reported
> that gap rather than hiding it, because a forecast that claims more confidence
> than it has is worse than one that's honestly uncertain.

### "Explain the bias-variance tradeoff using an example from your project."

> Persistence, my baseline, has high bias, it's too simple to capture real
> growth, but low variance, it never overreacts to noise. Undamped log-linear
> drift is the opposite, low bias when growth is steady but high variance, it
> extrapolates any short-term wobble into an extreme long-run trend, and it
> actually scores worse than the baseline at 14 days because of that. The
> damped renewal model sits in between: it captures real epidemic mechanics but
> pulls its own growth rate back toward 1 over time, which is a built-in
> variance reduction. That's the model that won.

---

## B4. Causal inference

### "How is your causal analysis different from just running a regression?"

> Several ways, but the important one is that I distinguish between an estimate
> being *robust* and an estimate being *identified*, and most regression-based
> analyses only check the first. Robust means the number doesn't move much when
> you stress-test it, different subsamples, different controls, placebo checks.
> Identified means your study design could, in principle, answer the causal
> question at all. My lockdown estimate is robust, it survives four separate
> stress tests. It is not identified, because I can show directly that
> government policy was decided by the same thing that decided the death count,
> the severity of the outbreak at that moment. A number can be perfectly robust
> and still be measuring the wrong direction of causation entirely, and proving
> that distinction was the actual point of the causal section, not just running
> the regression.

### "What is a confounder, and how did you handle it?"

> A confounder is a hidden factor that causes both the thing you're calling the
> treatment and the thing you're calling the outcome, which makes them move
> together even with no direct link between them. I handled it by writing down
> an explicit causal diagram first, with arrows for every relationship I
> believe exists, then using Pearl's back-door criterion to mechanically derive
> which variables need to be controlled for. That criterion is implemented in
> code and unit tested, so the adjustment set is a logical consequence of the
> diagram rather than a judgment call I made on the spot.

### "What is a mediator and why is adjusting for one a mistake?"

> A mediator is something your treatment causes, that then goes on to affect
> the outcome. If lockdowns reduce social contact, and reduced social contact
> reduces deaths, social contact is a mediator sitting between lockdown and
> deaths. If you control for it, you're statistically holding it fixed while
> asking about the effect of lockdown, which removes exactly the pathway
> lockdown works through. It's like asking whether exercise improves health
> while holding fitness constant, you've deleted the mechanism you're trying to
> measure. My causal graph deliberately includes several known mediators, like
> testing volume and vaccination speed, specifically so the back-door criterion
> can refuse them, and I have a test that confirms the criterion rejects every
> one of them.

### "Your effect estimate passed every robustness check. Why isn't that enough?"

> Because robustness checks and identification are answering different
> questions. A robustness check asks: does this number stay roughly the same
> under stress? An identification check asks: could my study design detect the
> true causal effect in the first place, or is it structurally blind to
> confounding? A number produced entirely by reverse causation can be perfectly
> stable. It'll survive a placebo test, because the reverse-causal relationship
> is really there in the data, not a fluke. It'll survive subsampling for the
> same reason. In my project, all four robustness checks pass and I then show
> directly, through a separate test on outbreak severity during the policy
> window, that the treatment itself is being decided by the outcome. That's why
> I built a separate identification test rather than stopping once the
> robustness checks passed.

### "What is Double Machine Learning and why not just use linear regression?"

> Ordinary regression assumes every confounder enters the outcome in a straight
> line, and that's false here. COVID mortality risk doesn't rise linearly with
> age, it rises steeply and non-linearly, so a linear age term leaves behind
> residual confounding that lands on your treatment estimate and biases it.
> Double ML fixes this by using flexible, non-linear models to separately
> predict the outcome and the treatment from the confounders, keeping only the
> parts neither prediction can explain, and then checking whether those two
> leftover parts move together. That lets the confounders be modeled flexibly
> while the final causal estimate stays interpretable. I validated this
> concretely: I simulated data where the confounding is deliberately
> non-linear, showed plain OLS gets a biased answer on it, and showed Double ML
> recovers the true planted effect to within about 0.06.

### "What's cross-fitting and why does it matter?"

> When you use a flexible model to predict something from the confounders, that
> model partly memorises its training data. If you then used its predictions on
> the same rows it was trained on, its errors would look artificially small,
> which biases the causal estimate. Cross-fitting splits the data into folds
> and only ever predicts a fold using a model trained on the other folds, so
> every prediction is genuinely out-of-sample. It's the same idea as a
> train-test split, just applied inside a single estimation procedure rather
> than as a one-time evaluation.

### "How did you check your causal finding wasn't a fluke?"

> Four separate stress tests, each with a specific prediction attached if the
> finding is real. Shuffle the treatment randomly across countries, and a real
> effect should vanish, mine dropped from 0.035 to about 0.001. Add a
> completely random, meaningless variable as an extra control, and a real
> effect shouldn't move, mine shifted by under 1%. Re-run on sixty random 80%
> subsamples, and a real effect should keep the same sign every time, mine did,
>100% of draws. Drop each continent one at a time and re-estimate, and a real
> effect shouldn't depend on any single region, mine stayed positive in every
> case. All four passed. And as I mentioned, passing those doesn't mean the
> effect is causal, it means the effect is stable, which is a necessary but not
> sufficient condition, and I have a separate section explaining exactly why it
> still isn't identified.

### "What is an E-value?"

> It answers: how strong would an unmeasured confounder need to be, in
> association with both my treatment and my outcome, to fully explain away my
> observed effect? A small E-value, close to 1, means a weak, plausible hidden
> factor could explain the whole thing, so the finding is fragile. A large one,
> 4 or more, means the hidden factor would have to be stronger than almost
> anything I've already measured, which is a much harder story to make up.
> Mine came out at 2.18, moderate, meaning a real but not enormous hidden
> confounder could still move the number, which is one more piece of evidence
> alongside the direct simultaneity test that this isn't a clean causal
> estimate.

### "What's a synthetic control and when would you use one?"

> It's for when one unit does something interesting and no single other unit is
> a fair comparison. Instead of picking one comparison region, you build a
> synthetic version of the treated region as a weighted blend of several
> untreated ones, choosing weights that make the blend match the treated
> region's history as closely as possible before the intervention. The weights
> are constrained to be non-negative and sum to one, so the synthetic
> counterfactual can never be an extrapolation, only ever a mix of things that
> actually happened. I used it to look at Maharashtra's April 2021 lockdown,
> and it returned a null result, and more importantly, the synthetic version
> had already stopped tracking the real Maharashtra three weeks before the
> lockdown even started, which tells you the design itself couldn't have
> answered the question, regardless of what number came out.

### "You got a null result from your synthetic control. How do you know that's not just a weak effect?"

> Because I checked the design's own validity before trusting the number. Two
> tests. First, I ran a placebo-in-time check, pretending the lockdown happened
> 45 days earlier than it really did, and that fake intervention produced a
> *bigger* apparent effect than the real one, which means the method is picking
> up ordinary drift between the regions, not anything policy-related. Second, I
> looked at the pre-period fit directly, and the synthetic Maharashtra had
> already diverged from the real one three weeks before the actual lockdown. A
> comparison that doesn't match beforehand can't tell you anything about what
> happened after. So I reported it as a failed attempt at identification, not
> as evidence the lockdown had no effect. Those are different claims, and
> mixing them up is a common mistake.

---

## B5. Software engineering

### "How did you make sure your pipeline is reproducible?"

> Four things, layered. Every random process in the pipeline is seeded from one
> constant, so re-running it gives identical numbers. Every downloaded file is
> hashed with SHA-256 and the hash, URL, size and download time are saved to a
> manifest, so any result traces back to the exact bytes that produced it.
> Every library dependency is pinned to a tested version range in
> requirements.txt. And the intermediate results at every stage get written to
> disk as parquet or CSV files, so a later stage never needs to silently
> recompute something using a different code path than what produced the
> original number.

### "Walk me through your testing strategy."

> 101 tests, and the design principle is that every one of them runs on
> synthetic data where I already know the correct answer, rather than on real
> downloaded data. That has two benefits: the whole suite runs in under a
> minute with no internet connection, and it can't silently stop testing
> anything just because a website changed its file format. The strongest tests
> simulate data from a known process and check the code recovers the known
> truth, for example I simulate an epidemic with reproduction number fixed at
> exactly 1.5 and assert the estimator recovers 1.5 to within 2%. I also have
> regression tests written after fixing real bugs, so those bugs can't silently
> come back.

### "Tell me about a bug you found and fixed."

> I had a line that read `learner or default_learner()`, which looks completely
> reasonable, use the learner if one was given, otherwise use a default. The
> problem is that Python evaluates the truthiness of the learner object to
> decide, and scikit-learn's models define truthiness by checking how many
> trees they contain, which raises an error on a model that hasn't been fit
> yet. So passing in a custom, not-yet-fitted learner crashed silently inside a
> try-except block, which made every single robustness check in my causal
> analysis return nothing, while the pipeline logged "0 of 4 checks passed" as
> though that were a legitimate result rather than a total failure. I fixed the
> check itself to `learner is None`, wrote a regression test that explicitly
> passes a custom learner, and separately added a check that raises an error
> whenever any part of that suite produces zero results, so an empty result set
> can never again quietly look like a clean pass.

### "Why do you cache downloaded data, and how did you make the caching safe?"

> Mainly speed and reliability, the main dataset is about 100 megabytes and
> re-downloading it every run would be slow and would also make the pipeline
> depend on an external server being up every single time. But the important
> part is making the cache safe against a failed download. I stream the
> download to a temporary file with a different extension, and only rename it
> to the real filename after the download completes successfully. If the
> connection drops partway through, you're left with an incomplete temp file
> that nothing mistakes for real data, rather than a file with the correct name
> that's silently missing half its content.

### "How would you design this pipeline to run at larger scale, say for 10x more countries or a live daily feed?"

> The current design already separates cleanly into independent stages, data
> ingestion, forensics scoring, forecasting, causal analysis, capacity
> modelling, and each one caches its own output, so scaling up mostly means
> those stages take proportionally longer rather than needing a redesign. For a
> live daily feed I'd change three things. First, the backtest currently
> recomputes from scratch, I'd want it to update incrementally, only scoring new
> forecast origins as they arrive rather than the whole history. Second, I'd
> move the Monte Carlo simulations in the capacity module, which are currently
> the slowest part, onto something that can parallelise across countries
> properly rather than looping. Third, I'd add alerting on the data-quality
> scorecard itself, since I already have machinery that detects reporting
> anomalies, running it daily and paging someone when a country's reliability
> score suddenly drops would be a natural extension.

### "What does 'causal' mean in the context of your code, as opposed to statistics?"

> In the software sense, a function is causal if its output at time i depends
> only on inputs from time i and earlier, never on the future. That's a
> completely different meaning from causal *inference*, and the fact that they
> share a word is actually a bit of a trap. In my project this software
> definition matters enormously for the forecasting backtest, because I
> precompute expensive features like the reproduction number once per country
> and then index into that array at every backtest origin, purely for speed.
> That shortcut is only safe if the computation truly never looks ahead, so I
> wrote a test that perturbs future values by a factor of 7 and checks the past
> output doesn't change at all. If that test ever failed, the entire backtest
> would be invalid, so it's one of the tests I'd call load-bearing.

---

## B6. Hard follow-ups

Interviewers sometimes push on the same point from a different angle to see if
you understand it or memorised it. Here are the harder versions.

### "If your lockdown effect isn't causal, why publish it at all?"

> Because the finding *is* the identification failure, not the coefficient
> itself. The number 0.035 isn't the interesting output of this section, the
> demonstration that a fully robustness-tested estimate can still be
> non-causal is. That's a genuinely useful thing to show, because it's exactly
> the mistake that gets made constantly in applied causal work: someone runs
> four robustness checks, all pass, and they report the number as an effect. I
> wanted to show both halves, here's an estimate that looks bulletproof by the
> usual standard, and here's the specific, checkable reason it still isn't one.

### "Couldn't you have just picked a better identification strategy from the start instead of showing the failure?"

> I could have gone straight for a design like regression discontinuity or an
> instrument, but I didn't have data that supports either of those cleanly at
> the country level, and pretending otherwise would have been worse than
> reporting the honest gap. Country-level lockdown timing is fundamentally
> tangled up with country-level severity, there's no natural experiment hiding
> in this particular dataset. What I did instead was name the specific design
> that would work, staggered policy rollout at the district level with matched
> pre-trends, which is a real, buildable next step, rather than forcing a
> technique the data can't actually support.

### "Your forecast model only beats the baseline when cases are falling. Isn't that a pretty weak result to headline?"

> It would be weak if I'd hidden it, but the point of reporting the breakdown by
> regime was specifically to not let the aggregate number hide it. The
> aggregate, 23% better at 7 days, is technically true and would be an easy
> thing to lead with, but it's mostly driven by the easy case, extrapolating a
> decline. I split results by whether cases were growing, flat, or falling
> specifically to find out whether that was true, and it was. I'd rather ship a
> forecast with an honestly narrow claim, it helps meaningfully when cases are
> declining, than an inflated one that implies uniform 23% improvement everywhere.

### "How do you know your synthetic control failure isn't just a mistake in your code?"

> Because I checked it two independent ways that would have caught a code bug
> differently. I ran a placebo-in-time test, faking the intervention date 45
> days earlier, expecting no effect if the code and design were sound, and
> instead got a bigger effect than the real one, which is a property of the
> data and comparison group, not something a code bug in the fitting procedure
> would produce that way. And separately I looked directly at the pre-period
> fit quality and saw the divergence happening visually, three weeks before the
> real intervention, which is a data pattern I can point to on a chart, not an
> arithmetic error. Those are two different failure modes that would need two
> different bugs to fake, which is why I trust the conclusion is about the
> data, not the implementation.

### "Isn't 101 tests on synthetic data missing the point? Shouldn't you test on real data too?"

> Synthetic tests and real-data validation are answering different questions,
> and I do both, just in different places. The synthetic tests check
> correctness, does the code compute what it claims to compute, and they need a
> known right answer, which real data essentially never gives you cleanly. The
> real-data check happens separately, in the ICU capacity model, where I
> compare model output against actual reported ICU occupancy for about 30
> countries and get a correlation of 0.94 before vaccination. That's the
> validation step, and it's deliberately kept apart from the unit tests, because
> mixing them would mean a website changing its data format could break my
> correctness tests, which is exactly the fragility I built the synthetic suite
> to avoid.

---

## B7. Questions to ask them

Asking a sharp question back signals you think like an engineer, not just that
you can answer questions. Pick two or three that fit the conversation.

- "When a model in production starts drifting, how do you find out, is it
  monitored automatically or does someone notice manually?"
- "How do you currently separate 'this number is statistically significant'
  from 'this number is safe to act on'? Is that distinction something the team
  actively discusses?"
- "What does your test suite depend on? Does it need a live database or
  external API, and how often does that cause flaky failures?"
- "When a data source changes its schema upstream, how do you find out before
  it silently breaks a downstream report?"
- "Is there a recent case where a robust-looking result turned out not to be
  causal once someone looked closer? What caught it?"

---

MIT licensed, part of [Yug Verma](https://github.com/Yugverma17)'s
[Pandemic Data Science](https://github.com/Yugverma17/Pandemic-Data-Science)
project.
