"""Published numbers of other conformal KGQA methods, for side-by-side tables and plots.

Source: CPR (Lin et al., arXiv 2605.08077), Table 2. UaG numbers are as reported by CPR
(UaG's own code was unavailable). ECR in %, APSS = average number of answer entities.
A missing alpha means the method did not report it or failed the coverage requirement.
"""

REFERENCE = {
    "webqsp": {
        "CPR": {0.3: (76.8, 17.77), 0.4: (67.3, 10.10), 0.5: (61.4, 7.14),
                0.6: (58.3, 5.45), 0.7: (55.1, 3.01), 0.8: (50.8, 1.62)},
        "UaG": {0.3: (72.3, 98.81), 0.4: (60.4, 32.52), 0.5: (51.8, 14.29),
                0.6: (41.8, 8.23), 0.7: (37.4, 5.58), 0.8: (25.6, 1.67)},
    },
    "cwq": {
        "CPR": {0.3: (76.6, 23.98), 0.4: (68.1, 15.28), 0.5: (58.8, 11.40),
                0.6: (50.2, 8.35), 0.7: (42.8, 6.08), 0.8: (29.9, 3.39)},
        "UaG": {0.6: (42.1, 120.20), 0.7: (37.3, 99.16), 0.8: (26.7, 27.96)},
    },
}


def reference_for(dataset: str) -> dict[str, dict[float, tuple[float, float]]]:
    """{method: {alpha: (ECR %, APSS)}}; empty if the dataset has no published numbers."""
    return REFERENCE.get(dataset.lower(), {})
