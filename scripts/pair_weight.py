"""
pair_weight.py -- the persistence pair weight of the article's equation (eq:pairweight), in one place.

Under the gradual-adjustment model (Appendix D) a manager keeps a fraction a of yesterday's risk direction
each day, so a share a^u of today's direction survives u days on; today's holdings correlation between two
managers is informative while both survive, (a_i a_j)^u, and an allocation held h days experiences the average:

    W_ij(h) = (1/h) sum_{u=1..h} (a_i a_j)^u = (a_i a_j) (1 - (a_i a_j)^h) / (h (1 - a_i a_j)).

a_i is read off the manager's own persistence curve phi_i(u) as the slope of log phi against u, fitted through
the origin (a long-only floor f_i is removed first when given). The geometric mean sqrt(phibar_i phibar_j) of the
horizon-average persistences is the fitting-free shortcut; it bounds W from above (Cauchy-Schwarz).

    retention(curve) -> a curve: phi(u) for u = 0..umax (index u), phi(0) = 1 allowed
    pair_weight(a_i, a_j, h) -> W_ij(h)
    weight_matrix(a, h) -> n x n matrix of W_ij(h) from a vector of retention rates
    geo_weight_matrix(phibar) -> n x n matrix of sqrt(phibar_i phibar_j) (the shortcut)
"""
import numpy as np


def fit_floor_geometric(curve, umax=None):
    """phi(u) = f + (1 - f) a^u fitted by least squares on the curve indexed by u (u = 0 is today, ignored):
    the floor f is the part of today's direction a book keeps for good (a long-only book's mean direction),
    a the daily retention of the rest. f is searched on a grid in [0, min phi) and a is the closed-form slope of
    log((phi - f)/(1 - f)) through the origin at each f; returns (a, f, r2) with r2 in levels. With f = 0 this is
    the plain geometric fit. A curve that never bends toward a floor over the measured horizons fits f near 0."""
    c = np.asarray(curve, float)
    u = np.arange(len(c), dtype=float)
    if umax is not None:
        c, u = c[:umax + 1], u[:umax + 1]
    ok = (u >= 1) & np.isfinite(c)
    c, u = c[ok], u[ok]
    # fit up to the first non-positive lag: a fast rotator decorrelates within the horizon and noise puts its tail at or just
    # below zero; rejecting the whole curve would leave the pair weight undefined on such windows. A curve positive at
    # every lag is fitted whole.
    bad = np.flatnonzero(c <= 0)
    if len(bad):
        c, u = c[:bad[0]], u[:bad[0]]
    if len(c) < 2:
        return np.nan, np.nan, np.nan
    best = (np.inf, np.nan, np.nan)
    for f in np.linspace(0.0, max(0.0, c.min() * 0.98), 50):
        y = (c - f) / (1.0 - f)
        if np.any(y <= 0):
            continue
        lam = -(np.log(y) * u).sum() / (u * u).sum()
        pred = f + (1.0 - f) * np.exp(-lam * u)
        sse = float(((c - pred) ** 2).sum())
        if sse < best[0]:
            best = (sse, float(np.exp(-lam)), float(f))
    sse, a, f = best
    sst = float(((c - c.mean()) ** 2).sum())
    return a, f, (1.0 - sse / sst if sst > 0 else np.nan)


def retention(curve, umax=None, floor="fit"):
    """Daily retention rate a of the decaying component of a persistence curve. floor='fit' fits the floor
    (default); a number removes that floor; 0 is the plain geometric fit through the origin."""
    if floor == "fit":
        return fit_floor_geometric(curve, umax)[0]
    c = np.asarray(curve, float)
    u = np.arange(len(c), dtype=float)
    if umax is not None:
        c, u = c[:umax + 1], u[:umax + 1]
    if floor:
        c = (c - floor) / (1.0 - floor)
    ok = (u >= 1) & np.isfinite(c) & (c > 0)
    if ok.sum() < 2:
        return np.nan
    lam = -(np.log(c[ok]) * u[ok]).sum() / (u[ok] ** 2).sum()
    return float(np.exp(-lam))


def retention_from_phibar(phibar, h):
    """Daily retention a solving (1/h) sum_{u=1..h} a^u = phibar, for callers that only have the horizon average
    (bisection; phibar of 1 gives 1, at or below 0 gives 0)."""
    if not np.isfinite(phibar):
        return np.nan
    if phibar >= 1.0:
        return 1.0
    if phibar <= 0.0:
        return 0.0
    u = np.arange(1, h + 1)
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if np.mean(mid ** u) < phibar:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pair_weight(a_i, a_j, h):
    """W_ij(h) = (1/h) sum_{u=1..h} (a_i a_j)^u, in closed form; 1 when both retain everything."""
    if not (np.isfinite(a_i) and np.isfinite(a_j)):
        return np.nan
    q = float(np.clip(a_i * a_j, 0.0, 1.0))
    if q >= 1.0:
        return 1.0
    if q <= 0.0:
        return 0.0
    return q * (1.0 - q ** h) / (h * (1.0 - q))


def pair_weight_stale(a_i, a_j, h, s):
    """A book that is s days old when used over the next h days: the survival on day u of the horizon is
    (a_i a_j)^(s+u), so the weight is (a_i a_j)^s W_ij(h) (the article: staleness moves along the same curve)."""
    w = pair_weight(a_i, a_j, h)
    if not np.isfinite(w):
        return np.nan
    q = float(np.clip(a_i * a_j, 0.0, 1.0))
    return (q ** s) * w


def weight_matrix(a, h, s=0):
    """n x n matrix of W_ij(h) from a vector of retention rates; s > 0 shifts every pair by a filing age of s days."""
    a = np.asarray(a, float); n = len(a)
    W = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            W[i, j] = pair_weight_stale(a[i], a[j], h, s) if s else pair_weight(a[i], a[j], h)
    return W


def geo_weight_matrix(phibar):
    p = np.clip(np.asarray(phibar, float), 0.0, None)
    return np.sqrt(np.outer(p, p))
