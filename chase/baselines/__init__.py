"""Same-scorer baselines: every method uses the trained CHASE scorer and differs only in how it builds the set.

topk     conformal top-K: rank nodes by sigma, return a calibrated number K of them
beam     CPR-style: fixed TreeG beam, then a post-hoc conformal threshold on the retrieved paths
per_hop  UaG-style: one conformal threshold per hop (and one for stopping), pruning inside the search
run      evaluate all of them (and CHASE) on one trained model
report   mean ± std over seeds
"""
