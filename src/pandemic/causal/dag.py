"""The causal graph and a working back-door criterion.

The graph is written down here and the adjustment set is checked against it with
Pearl's back-door criterion (Pearl 1995, Biometrika 82(4):669-688):

    Z is admissible for estimating the effect of T on Y if
      (i)  no member of Z is a descendant of T, and
      (ii) Z d-separates T from Y in the graph with all edges out of T removed.

Condition (i) rules out controlling for a mediator. Vaccination rate, testing
volume and observed case counts are all consequences of the policy under
evaluation, so conditioning on them deletes part of the effect being estimated
and biases it toward zero. Having the graph in code makes that a check rather
than a matter of taste.
"""

from __future__ import annotations

import networkx as nx

TREATMENT = "stringency"
OUTCOME = "deaths"

# Edges are (cause, effect). Each is a substantive claim, and every one of them
# is arguable -- which is the point of publishing the graph rather than a
# variable list: a reader who disagrees can say precisely where.
EDGES: list[tuple[str, str]] = [
    # Demography and wealth drive both the response and the death toll.
    ("median_age", "deaths"),
    ("median_age", "stringency"),
    ("wealth", "stringency"),
    ("wealth", "deaths"),
    ("wealth", "health_capacity"),
    ("wealth", "testing"),
    ("health_capacity", "deaths"),
    ("comorbidity", "deaths"),
    ("wealth", "comorbidity"),
    ("density", "transmission"),
    ("density", "stringency"),
    ("transmission", "deaths"),
    ("stringency", "transmission"),
    # Epidemic timing: countries seeded later had more warning, and warning
    # changes both how fast they acted and how bad the outbreak got.
    ("seeding_delay", "stringency"),
    ("seeding_delay", "deaths"),
    ("onset_growth", "stringency"),   # governments react to what they observe
    ("onset_growth", "transmission"),
    # Post-treatment consequences. Present in the graph precisely so the
    # criterion can refuse them.
    ("stringency", "testing"),
    ("testing", "observed_cases"),
    ("transmission", "observed_cases"),
    ("stringency", "vaccination_speed"),
    ("vaccination_speed", "deaths"),
]

# Which measured column stands in for each graph node.
NODE_TO_COLUMN = {
    "median_age": ["median_age", "aged_65_older", "life_expectancy"],
    "wealth": ["log_gdp_per_capita", "human_development_index"],
    "health_capacity": ["hospital_beds_per_thousand"],
    "comorbidity": ["diabetes_prevalence", "cardiovasc_death_rate"],
    "density": ["log_population_density", "log_population"],
    "seeding_delay": ["seeding_delay_days"],
    "onset_growth": ["onset_growth_rate"],
}


def build_dag(edges: list[tuple[str, str]] | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(edges or EDGES)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("the causal graph contains a cycle")
    return g


def _d_separated(g: nx.DiGraph, x: set[str], y: set[str], z: set[str]) -> bool:
    """d-separation, tolerant of the networkx rename in 3.3."""
    if hasattr(nx, "is_d_separator"):
        return bool(nx.is_d_separator(g, x, y, z))
    return bool(nx.d_separated(g, x, y, z))  # networkx < 3.3


def is_valid_backdoor(g: nx.DiGraph, treatment: str, outcome: str,
                      adjustment: set[str]) -> tuple[bool, str]:
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


def minimal_backdoor_sets(g: nx.DiGraph, treatment: str = TREATMENT,
                          outcome: str = OUTCOME,
                          max_size: int = 5) -> list[set[str]]:
    """Enumerate minimal admissible adjustment sets, smallest first.

    Brute force over subsets is fine at this graph size and has the advantage of
    being obviously correct. Minimality matters: adjusting for more than
    necessary costs precision and can open a collider path.
    """
    from itertools import combinations

    candidates = [n for n in g.nodes
                  if n not in {treatment, outcome}
                  and n not in nx.descendants(g, treatment)]

    found: list[set[str]] = []
    for size in range(max_size + 1):
        for combo in combinations(sorted(candidates), size):
            s = set(combo)
            if any(prev <= s for prev in found):
                continue  # a subset already works, so this one is not minimal
            ok, _ = is_valid_backdoor(g, treatment, outcome, s)
            if ok:
                found.append(s)
    return found


def columns_for(nodes: set[str]) -> list[str]:
    """Map graph nodes onto the measured columns that proxy them."""
    cols: list[str] = []
    for n in sorted(nodes):
        cols.extend(NODE_TO_COLUMN.get(n, []))
    return list(dict.fromkeys(cols))


def describe() -> dict:
    """Summary of the graph and its implications, for the report."""
    g = build_dag()
    minimal = minimal_backdoor_sets(g)
    mediators = sorted(nx.descendants(g, TREATMENT) - {OUTCOME})
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "mediators_excluded": mediators,
        "minimal_adjustment_sets": [sorted(s) for s in minimal[:5]],
        "chosen_adjustment_columns": columns_for(minimal[0]) if minimal else [],
    }
