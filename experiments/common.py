"""Shared synthetic-instance generation and independent ctFMNL objective oracle."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Iterable




@dataclass(frozen=True)
class OracleProduct:
    product_id: object
    a: float
    m: float


class ObjectiveOracle:
    """Independent float64 evaluator for the ctFMNL objective.

    Focal membership is a model branch, not a numerical approximation: the
    exact condition ``(1+w)*a >= amax`` is used without an epsilon margin.
    """

    def __init__(self, products: Iterable, a0: float, w: float, delta: float):
        self.a0 = float(a0)
        self.w = float(w)
        self.delta = float(delta)
        if not (math.isfinite(self.a0) and self.a0 > 0
                and math.isfinite(self.w) and self.w >= 0
                and math.isfinite(self.delta) and self.delta >= 0):
            raise ValueError("require finite a0 > 0, w >= 0, delta >= 0")
        self.rows = []
        self.index = {}
        for p in products:
            product_id = getattr(p, "productId", getattr(p, "product_id", None))
            a = float(p.a)
            m = float(p.m)
            if product_id is None or product_id in self.index:
                raise ValueError("product IDs must be present and unique")
            if not (math.isfinite(a) and a > 0
                    and math.isfinite(m) and m >= 0):
                raise ValueError("products require finite a > 0 and m >= 0")
            row = OracleProduct(product_id, a, m)
            self.index[product_id] = len(self.rows)
            self.rows.append(row)

    def __call__(self, selected) -> float:
        ids = list(selected)
        if len(ids) != len(set(ids)):
            raise ValueError("assortment contains duplicate product IDs")
        try:
            rows = [self.rows[self.index[i]] for i in ids]
        except KeyError as exc:
            raise ValueError(f"unknown product ID: {exc.args[0]!r}") from exc
        amax = max([self.a0, *(p.a for p in rows)])
        outside_focal = (1.0 + self.w) * self.a0 >= amax
        numerator = 0.0
        denominator = self.a0 * (1.0 + self.delta * outside_focal)
        for p in rows:
            focal = (1.0 + self.w) * p.a >= amax
            weight = 1.0 + self.delta * focal
            numerator += p.m * p.a * weight
            denominator += p.a * weight
        if denominator <= 0.0:
            raise FloatingPointError("nonpositive ctFMNL denominator")
        return numerator / denominator


def objective(products, selected, a0: float, w: float, delta: float) -> float:
    return ObjectiveOracle(products, a0, w, delta)(selected)


ABS_TOL = 1e-10
REL_TOL = 1e-8


def close_objectives(x: float, y: float) -> bool:
    """Shared absolute-plus-relative tolerance for objective comparisons."""
    return abs(x - y) <= ABS_TOL + REL_TOL * max(abs(x), abs(y))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file using bounded-memory reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
