"""
GeoTechHub Bearing Capacity Calculation Engine
V1 production-style engineering calculator for shallow foundations.

Units expected by this engine:
- Length: ft
- Force: kip for structural loads
- Moment: kip-ft
- Unit weight: pcf
- Cohesion/pressure: psf
- Soil elastic modulus: ksf
- Settlement: inches in output

Engineering note:
This module implements classical bearing capacity relationships with simplified
shape, depth, and load inclination factors suitable for preliminary design and
screening. Final design should be reviewed by a licensed geotechnical engineer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from math import atan, exp, inf, pi, radians, tan, sqrt
from typing import Any, Dict, List, Tuple


EPS = 1.0e-9


@dataclass
class BearingCapacityInput:
    project_name: str = "GeoTechHub Bearing Capacity Check"
    method: str = "Meyerhof"  # Terzaghi, Meyerhof, Hansen, Vesic
    footing_shape: str = "rectangular"  # strip, square, circular, rectangular

    B_ft: float = 6.0
    L_ft: float = 10.0
    Df_ft: float = 3.0

    P_kip: float = 250.0
    Mx_kipft: float = 0.0  # moment about x-axis; pressure varies along L direction
    My_kipft: float = 0.0  # moment about y-axis; pressure varies along B direction
    Hx_kip: float = 0.0
    Hy_kip: float = 0.0

    cohesion_psf: float = 500.0
    phi_deg: float = 30.0
    gamma_pcf: float = 120.0
    gamma_sat_pcf: float = 125.0
    water_table_depth_ft: float = 999.0

    FS_bearing_required: float = 3.0
    FS_sliding_required: float = 1.5
    FS_overturning_required: float = 2.0

    base_friction_angle_deg: float = 25.0
    base_adhesion_psf: float = 0.0
    passive_resistance_kip: float = 0.0
    include_passive_resistance: bool = False

    elastic_modulus_ksf: float = 500.0
    poisson_ratio: float = 0.35
    settlement_influence_factor: float = 1.0
    allowable_settlement_in: float = 1.0


# ---------------------------------------------------------------------------
# Validation and utility functions
# ---------------------------------------------------------------------------


def _clean_method(method: str) -> str:
    m = (method or "Meyerhof").strip().lower()
    aliases = {
        "terzaghi": "Terzaghi",
        "meyerhof": "Meyerhof",
        "hansen": "Hansen",
        "vesic": "Vesic",
        "vesić": "Vesic",
    }
    if m not in aliases:
        raise ValueError("method must be Terzaghi, Meyerhof, Hansen, or Vesic")
    return aliases[m]


def _clean_shape(shape: str) -> str:
    s = (shape or "rectangular").strip().lower()
    aliases = {
        "strip": "strip",
        "continuous": "strip",
        "square": "square",
        "rectangular": "rectangular",
        "rectangle": "rectangular",
        "circular": "circular",
        "circle": "circular",
    }
    if s not in aliases:
        raise ValueError("footing_shape must be strip, square, rectangular, or circular")
    return aliases[s]


def validate_input(inp: BearingCapacityInput) -> None:
    if inp.B_ft <= 0:
        raise ValueError("B_ft must be greater than zero")
    if inp.L_ft <= 0:
        raise ValueError("L_ft must be greater than zero")
    if inp.Df_ft < 0:
        raise ValueError("Df_ft cannot be negative")
    if inp.P_kip <= 0:
        raise ValueError("P_kip must be greater than zero")
    if inp.cohesion_psf < 0:
        raise ValueError("cohesion_psf cannot be negative")
    if not (0 <= inp.phi_deg < 50):
        raise ValueError("phi_deg should be between 0 and less than 50 degrees")
    if inp.gamma_pcf <= 0:
        raise ValueError("gamma_pcf must be greater than zero")
    if inp.FS_bearing_required <= 0:
        raise ValueError("FS_bearing_required must be greater than zero")
    if inp.elastic_modulus_ksf <= 0:
        raise ValueError("elastic_modulus_ksf must be greater than zero")
    if not (0 <= inp.poisson_ratio < 0.5):
        raise ValueError("poisson_ratio should be between 0 and less than 0.5")


def effective_unit_weight(inp: BearingCapacityInput) -> Dict[str, float]:
    """Return surcharge unit weight and bearing-term unit weight in pcf.

    Simplified water-table correction:
    - If water table is at or above foundation base, use submerged unit weight for
      the N_gamma term.
    - If water table is below foundation base but within B below base, linearly
      transition from submerged to moist unit weight.
    - Otherwise use moist unit weight.
    Surcharge q uses moist unit weight above base unless water table is within Df;
    then a weighted average is used.
    """
    gamma = inp.gamma_pcf
    gamma_sub = max(inp.gamma_sat_pcf - 62.4, 1.0)
    wt = inp.water_table_depth_ft
    Df = inp.Df_ft
    B = inp.B_ft

    if wt <= 0:
        gamma_surcharge = gamma_sub
    elif wt < Df:
        gamma_surcharge = (gamma * wt + gamma_sub * (Df - wt)) / max(Df, EPS)
    else:
        gamma_surcharge = gamma

    if wt <= Df:
        gamma_bearing = gamma_sub
    elif wt < Df + B:
        ratio = (wt - Df) / max(B, EPS)
        gamma_bearing = gamma_sub + ratio * (gamma - gamma_sub)
    else:
        gamma_bearing = gamma

    return {
        "gamma_surcharge_pcf": gamma_surcharge,
        "gamma_bearing_pcf": gamma_bearing,
        "gamma_submerged_pcf": gamma_sub,
    }


# ---------------------------------------------------------------------------
# Bearing capacity factors
# ---------------------------------------------------------------------------


def bearing_capacity_factors(phi_deg: float, method: str) -> Dict[str, float]:
    method = _clean_method(method)
    phi = radians(phi_deg)

    if abs(phi_deg) < 1.0e-7:
        return {"Nc": 5.14, "Nq": 1.0, "Ngamma": 0.0}

    Nq = exp(pi * tan(phi)) * tan(radians(45.0) + phi / 2.0) ** 2
    Nc = (Nq - 1.0) / tan(phi)

    if method == "Terzaghi":
        Ngamma = 1.5 * (Nq - 1.0) * tan(phi)
    elif method == "Meyerhof":
        Ngamma = (Nq - 1.0) * tan(1.4 * phi)
    elif method == "Hansen":
        Ngamma = 1.5 * (Nq - 1.0) * tan(phi)
    else:  # Vesic
        Ngamma = 2.0 * (Nq + 1.0) * tan(phi)

    return {"Nc": Nc, "Nq": Nq, "Ngamma": max(Ngamma, 0.0)}


def shape_factors(inp: BearingCapacityInput, factors: Dict[str, float], use_effective: bool) -> Dict[str, float]:
    shape = _clean_shape(inp.footing_shape)
    method = _clean_method(inp.method)
    dims = effective_footing_dimensions(inp) if use_effective else gross_footing_dimensions(inp)
    B = dims["B_check_ft"]
    L = dims["L_check_ft"]
    ratio = min(B, L) / max(B, L)
    phi = radians(inp.phi_deg)
    Nc = max(factors["Nc"], EPS)
    Nq = max(factors["Nq"], EPS)

    if shape == "strip":
        return {"sc": 1.0, "sq": 1.0, "sgamma": 1.0}

    if method == "Terzaghi":
        if shape == "square":
            return {"sc": 1.3, "sq": 1.0, "sgamma": 0.8}
        if shape == "circular":
            return {"sc": 1.3, "sq": 1.0, "sgamma": 0.6}
        return {"sc": 1.0 + 0.3 * ratio, "sq": 1.0, "sgamma": 1.0 - 0.2 * ratio}

    if method == "Meyerhof":
        tan45 = tan(radians(45.0) + phi / 2.0)
        return {
            "sc": 1.0 + 0.2 * ratio * tan45 ** 2,
            "sq": 1.0 + 0.1 * ratio * tan45 ** 2,
            "sgamma": max(1.0 - 0.4 * ratio, 0.6),
        }

    # Hansen/Vesic common practical form
    if abs(inp.phi_deg) < 1.0e-7:
        sc = 1.0 + 0.2 * ratio
        sq = 1.0
    else:
        sc = 1.0 + (Nq / Nc) * ratio
        sq = 1.0 + ratio * tan(phi)
    sgamma = max(1.0 - 0.4 * ratio, 0.6)
    return {"sc": sc, "sq": sq, "sgamma": sgamma}


def depth_factors(inp: BearingCapacityInput, factors: Dict[str, float], use_effective: bool) -> Dict[str, float]:
    method = _clean_method(inp.method)
    if method == "Terzaghi":
        return {"dc": 1.0, "dq": 1.0, "dgamma": 1.0}

    dims = effective_footing_dimensions(inp) if use_effective else gross_footing_dimensions(inp)
    B = max(dims["B_check_ft"], EPS)
    Df_over_B = inp.Df_ft / B
    phi = radians(inp.phi_deg)

    if Df_over_B <= 1.0:
        k = Df_over_B
    else:
        k = atan(Df_over_B)

    if abs(inp.phi_deg) < 1.0e-7:
        dc = 1.0 + 0.4 * k
        dq = 1.0
    else:
        dc = 1.0 + 0.2 * k * sqrt(max(factors["Nq"], 0.0))
        dq = 1.0 + 2.0 * tan(phi) * (1.0 - sin_phi(inp.phi_deg)) ** 2 * k

    return {"dc": dc, "dq": dq, "dgamma": 1.0}


def inclination_factors(inp: BearingCapacityInput) -> Dict[str, float]:
    method = _clean_method(inp.method)
    if method == "Terzaghi":
        return {"ic": 1.0, "iq": 1.0, "igamma": 1.0}

    H = sqrt(inp.Hx_kip ** 2 + inp.Hy_kip ** 2)
    V = max(inp.P_kip, EPS)
    ratio = min(max(H / V, 0.0), 0.95)
    i = max((1.0 - ratio) ** 2, 0.0)
    return {"ic": i, "iq": i, "igamma": i}


def sin_phi(phi_deg: float) -> float:
    from math import sin

    return sin(radians(phi_deg))


# ---------------------------------------------------------------------------
# Footing geometry, eccentricity, pressure, stability, settlement
# ---------------------------------------------------------------------------


def gross_footing_dimensions(inp: BearingCapacityInput) -> Dict[str, float]:
    shape = _clean_shape(inp.footing_shape)
    if shape == "strip":
        # Treat user-entered L as design strip length for load-pressure calculations.
        # For bearing capacity factors, strip behavior is controlled by B.
        pass
    if shape == "square":
        B = inp.B_ft
        L = inp.B_ft
    elif shape == "circular":
        # B is diameter. Use equivalent square dimensions for pressure grid convenience.
        B = inp.B_ft
        L = inp.B_ft
    else:
        B = inp.B_ft
        L = inp.L_ft
    return {"B_check_ft": B, "L_check_ft": L, "area_ft2": B * L}


def effective_footing_dimensions(inp: BearingCapacityInput) -> Dict[str, Any]:
    gross = gross_footing_dimensions(inp)
    B = gross["B_check_ft"]
    L = gross["L_check_ft"]
    P = max(inp.P_kip, EPS)
    ex = abs(inp.My_kipft) / P
    ey = abs(inp.Mx_kipft) / P
    B_eff = B - 2.0 * ex
    L_eff = L - 2.0 * ey
    valid = B_eff > 0 and L_eff > 0
    B_eff_safe = max(B_eff, EPS)
    L_eff_safe = max(L_eff, EPS)
    middle_third_x = ex <= B / 6.0 + EPS
    middle_third_y = ey <= L / 6.0 + EPS

    return {
        "B_gross_ft": B,
        "L_gross_ft": L,
        "A_gross_ft2": B * L,
        "ex_ft": ex,
        "ey_ft": ey,
        "B_eff_ft": B_eff,
        "L_eff_ft": L_eff,
        "A_eff_ft2": B_eff_safe * L_eff_safe if valid else 0.0,
        "B_check_ft": B_eff_safe,
        "L_check_ft": L_eff_safe,
        "valid_effective_area": valid,
        "middle_third_x_status": "PASS" if middle_third_x else "WARNING",
        "middle_third_y_status": "PASS" if middle_third_y else "WARNING",
        "middle_third_status": "PASS" if middle_third_x and middle_third_y else "WARNING",
    }


def bearing_capacity(inp: BearingCapacityInput) -> Dict[str, Any]:
    method = _clean_method(inp.method)
    validate_input(inp)

    eff = effective_footing_dimensions(inp)
    use_effective = eff["valid_effective_area"]
    factors = bearing_capacity_factors(inp.phi_deg, method)
    sf = shape_factors(inp, factors, use_effective=use_effective)
    df = depth_factors(inp, factors, use_effective=use_effective)
    infactors = inclination_factors(inp)
    gamma_info = effective_unit_weight(inp)

    B_used = eff["B_check_ft"] if use_effective else gross_footing_dimensions(inp)["B_check_ft"]
    q_surcharge = gamma_info["gamma_surcharge_pcf"] * inp.Df_ft
    gamma_bearing = gamma_info["gamma_bearing_pcf"]

    c_term = inp.cohesion_psf * factors["Nc"] * sf["sc"] * df["dc"] * infactors["ic"]
    q_term = q_surcharge * factors["Nq"] * sf["sq"] * df["dq"] * infactors["iq"]
    gamma_term = 0.5 * gamma_bearing * B_used * factors["Ngamma"] * sf["sgamma"] * df["dgamma"] * infactors["igamma"]

    q_ult_gross = c_term + q_term + gamma_term
    q_ult_net = max(q_ult_gross - q_surcharge, 0.0)
    q_allow_gross = q_ult_gross / inp.FS_bearing_required
    q_allow_net = q_ult_net / inp.FS_bearing_required

    q_applied = applied_pressures(inp)
    q_eff_psf = inp.P_kip * 1000.0 / max(eff["A_eff_ft2"], EPS) if eff["valid_effective_area"] else inf
    q_net_eff_psf = max(q_eff_psf - q_surcharge, 0.0) if q_eff_psf != inf else inf

    bearing_status = "PASS" if q_eff_psf <= q_allow_gross and eff["valid_effective_area"] else "FAIL"

    return {
        "method": method,
        "factors": round_dict(factors),
        "shape_factors": round_dict(sf),
        "depth_factors": round_dict(df),
        "inclination_factors": round_dict(infactors),
        "gamma_correction": round_dict(gamma_info),
        "surcharge_q_psf": round(q_surcharge, 2),
        "c_term_psf": round(c_term, 2),
        "q_term_psf": round(q_term, 2),
        "gamma_term_psf": round(gamma_term, 2),
        "qult_gross_psf": round(q_ult_gross, 2),
        "qult_net_psf": round(q_ult_net, 2),
        "qallow_gross_psf": round(q_allow_gross, 2),
        "qallow_net_psf": round(q_allow_net, 2),
        "q_effective_gross_psf": round(q_eff_psf, 2) if q_eff_psf != inf else "Infinity",
        "q_effective_net_psf": round(q_net_eff_psf, 2) if q_net_eff_psf != inf else "Infinity",
        "bearing_FS_actual_gross": round(q_ult_gross / max(q_eff_psf, EPS), 3) if q_eff_psf != inf else 0.0,
        "bearing_status": bearing_status,
        "contact_pressure_reference": q_applied,
    }


def applied_pressures(inp: BearingCapacityInput) -> Dict[str, Any]:
    dims = gross_footing_dimensions(inp)
    B = dims["B_check_ft"]
    L = dims["L_check_ft"]
    A = B * L
    P = inp.P_kip
    Mx = inp.Mx_kipft
    My = inp.My_kipft

    Ix = B * L ** 3 / 12.0
    Iy = L * B ** 3 / 12.0
    q_avg_ksf = P / max(A, EPS)

    corners = []
    q_values = []
    for x_label, x in [("left", -B / 2.0), ("right", B / 2.0)]:
        for y_label, y in [("bottom", -L / 2.0), ("top", L / 2.0)]:
            q_ksf = q_avg_ksf + Mx * y / max(Ix, EPS) + My * x / max(Iy, EPS)
            item = {
                "corner": f"{x_label}-{y_label}",
                "x_ft": round(x, 4),
                "y_ft": round(y, 4),
                "q_psf": round(q_ksf * 1000.0, 2),
            }
            corners.append(item)
            q_values.append(q_ksf * 1000.0)

    q_min = min(q_values)
    q_max = max(q_values)

    return {
        "q_avg_psf": round(q_avg_ksf * 1000.0, 2),
        "q_min_psf": round(q_min, 2),
        "q_max_psf": round(q_max, 2),
        "corners": corners,
        "full_compression_status": "PASS" if q_min >= -EPS else "WARNING",
        "contact_condition": "Full compression" if q_min >= -EPS else "Partial bearing / uplift predicted",
        "Ix_ft4": round(Ix, 4),
        "Iy_ft4": round(Iy, 4),
    }


def pressure_grid(inp: BearingCapacityInput, n: int = 31) -> Dict[str, Any]:
    dims = gross_footing_dimensions(inp)
    B = dims["B_check_ft"]
    L = dims["L_check_ft"]
    A = B * L
    P = inp.P_kip
    Mx = inp.Mx_kipft
    My = inp.My_kipft
    Ix = B * L ** 3 / 12.0
    Iy = L * B ** 3 / 12.0
    q_avg_ksf = P / max(A, EPS)

    xs = [(-B / 2.0) + i * B / (n - 1) for i in range(n)]
    ys = [(-L / 2.0) + j * L / (n - 1) for j in range(n)]
    z: List[List[float]] = []
    for y in ys:
        row = []
        for x in xs:
            q_psf = (q_avg_ksf + Mx * y / max(Ix, EPS) + My * x / max(Iy, EPS)) * 1000.0
            row.append(round(q_psf, 2))
        z.append(row)
    return {
        "x_ft": [round(v, 3) for v in xs],
        "y_ft": [round(v, 3) for v in ys],
        "q_psf_grid": z,
    }


def sliding_check(inp: BearingCapacityInput) -> Dict[str, Any]:
    H = sqrt(inp.Hx_kip ** 2 + inp.Hy_kip ** 2)
    dims = gross_footing_dimensions(inp)
    A = dims["area_ft2"]
    friction = inp.P_kip * tan(radians(inp.base_friction_angle_deg))
    adhesion = inp.base_adhesion_psf * A / 1000.0
    passive = inp.passive_resistance_kip if inp.include_passive_resistance else 0.0
    resistance = friction + adhesion + passive
    FS = inf if H <= EPS else resistance / H
    status = "PASS" if FS >= inp.FS_sliding_required else "FAIL"
    if H <= EPS:
        status = "PASS"
    return {
        "H_resultant_kip": round(H, 3),
        "friction_resistance_kip": round(friction, 3),
        "adhesion_resistance_kip": round(adhesion, 3),
        "passive_resistance_used_kip": round(passive, 3),
        "total_resistance_kip": round(resistance, 3),
        "FS_sliding": "Infinity" if FS == inf else round(FS, 3),
        "required_FS": inp.FS_sliding_required,
        "status": status,
    }


def overturning_check(inp: BearingCapacityInput) -> Dict[str, Any]:
    dims = gross_footing_dimensions(inp)
    B = dims["B_check_ft"]
    L = dims["L_check_ft"]
    P = inp.P_kip
    Mx = abs(inp.Mx_kipft)
    My = abs(inp.My_kipft)

    resisting_x = P * L / 2.0
    resisting_y = P * B / 2.0
    FSx = inf if Mx <= EPS else resisting_x / Mx
    FSy = inf if My <= EPS else resisting_y / My
    status_x = "PASS" if FSx >= inp.FS_overturning_required else "FAIL"
    status_y = "PASS" if FSy >= inp.FS_overturning_required else "FAIL"

    return {
        "resisting_moment_about_x_kipft": round(resisting_x, 3),
        "resisting_moment_about_y_kipft": round(resisting_y, 3),
        "overturning_moment_x_kipft": round(Mx, 3),
        "overturning_moment_y_kipft": round(My, 3),
        "FS_overturning_x": "Infinity" if FSx == inf else round(FSx, 3),
        "FS_overturning_y": "Infinity" if FSy == inf else round(FSy, 3),
        "required_FS": inp.FS_overturning_required,
        "status_x": status_x,
        "status_y": status_y,
        "status": "PASS" if status_x == "PASS" and status_y == "PASS" else "FAIL",
    }


def settlement_estimate(inp: BearingCapacityInput) -> Dict[str, Any]:
    eff = effective_footing_dimensions(inp)
    gamma_info = effective_unit_weight(inp)
    surcharge = gamma_info["gamma_surcharge_pcf"] * inp.Df_ft
    A_eff = max(eff["A_eff_ft2"], EPS)
    q_gross = inp.P_kip * 1000.0 / A_eff if eff["valid_effective_area"] else inf
    q_net = max(q_gross - surcharge, 0.0) if q_gross != inf else inf
    B = max(eff["B_check_ft"], EPS)
    Es_psf = inp.elastic_modulus_ksf * 1000.0
    settlement_ft = q_net * B * (1.0 - inp.poisson_ratio ** 2) * inp.settlement_influence_factor / Es_psf if q_net != inf else inf
    settlement_in = settlement_ft * 12.0 if settlement_ft != inf else inf
    status = "PASS" if settlement_in <= inp.allowable_settlement_in else "FAIL"
    if settlement_in == inf:
        status = "FAIL"
    return {
        "q_net_for_settlement_psf": round(q_net, 2) if q_net != inf else "Infinity",
        "elastic_modulus_ksf": inp.elastic_modulus_ksf,
        "poisson_ratio": inp.poisson_ratio,
        "influence_factor": inp.settlement_influence_factor,
        "estimated_settlement_in": round(settlement_in, 3) if settlement_in != inf else "Infinity",
        "allowable_settlement_in": inp.allowable_settlement_in,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Full calculation, assumptions, formulas
# ---------------------------------------------------------------------------


def calculate_bearing_capacity(data: Dict[str, Any] | BearingCapacityInput) -> Dict[str, Any]:
    if isinstance(data, BearingCapacityInput):
        inp = data
    else:
        inp = BearingCapacityInput(**data)

    inp.method = _clean_method(inp.method)
    inp.footing_shape = _clean_shape(inp.footing_shape)
    validate_input(inp)

    eff = effective_footing_dimensions(inp)
    contact = applied_pressures(inp)
    bearing = bearing_capacity(inp)
    sliding = sliding_check(inp)
    overturning = overturning_check(inp)
    settlement = settlement_estimate(inp)
    grid = pressure_grid(inp)

    statuses = {
        "bearing": bearing["bearing_status"],
        "effective_area": "PASS" if eff["valid_effective_area"] else "FAIL",
        "contact_full_compression": contact["full_compression_status"],
        "sliding": sliding["status"],
        "overturning": overturning["status"],
        "settlement": settlement["status"],
    }

    hard_fail = any(v == "FAIL" for v in statuses.values())
    warning = any(v == "WARNING" for v in statuses.values())
    if hard_fail:
        overall = "FAIL"
    elif warning:
        overall = "WARNING"
    else:
        overall = "PASS"

    governing = governing_message(statuses, bearing, contact, settlement, eff)

    return {
        "project_name": inp.project_name,
        "input_summary": asdict(inp),
        "effective_footing": round_nested(eff),
        "bearing_capacity": bearing,
        "contact_pressure": contact,
        "pressure_grid": grid,
        "sliding": sliding,
        "overturning": overturning,
        "settlement": settlement,
        "pass_fail": {
            "overall_status": overall,
            "governing_check": governing,
            "checks": statuses,
        },
        "engineering_assumptions": engineering_assumptions(inp),
        "formula_sheet": formula_sheet(),
    }


def governing_message(statuses: Dict[str, str], bearing: Dict[str, Any], contact: Dict[str, Any], settlement: Dict[str, Any], eff: Dict[str, Any]) -> str:
    if statuses["effective_area"] == "FAIL":
        return "Resultant eccentricity is outside the footing kern; effective area is zero or negative. Increase footing size or reduce moment."
    if statuses["bearing"] == "FAIL":
        return "Applied effective bearing pressure exceeds allowable gross bearing pressure."
    if statuses["contact_full_compression"] == "WARNING":
        return "Minimum contact pressure is negative; partial bearing/uplift is predicted. Effective area method controls bearing evaluation."
    if statuses["sliding"] == "FAIL":
        return "Sliding factor of safety is below the required value."
    if statuses["overturning"] == "FAIL":
        return "Overturning factor of safety is below the required value."
    if statuses["settlement"] == "FAIL":
        return "Estimated elastic settlement exceeds allowable settlement."
    return "All V1 design checks satisfy the selected criteria."


def engineering_assumptions(inp: BearingCapacityInput) -> List[str]:
    return [
        "Classical shallow foundation bearing capacity theory is used for preliminary design screening.",
        "Footing base is assumed level and bearing on a homogeneous soil layer unless otherwise evaluated by the designer.",
        "Water-table correction uses a simplified effective unit weight approach for surcharge and N_gamma bearing terms.",
        "Biaxial moment contact pressure is computed using rigid footing elastic pressure distribution.",
        "When uplift is predicted, effective footing area B' = B - 2e_x and L' = L - 2e_y is used for bearing evaluation.",
        "Sliding resistance includes base friction, optional base adhesion, and optional passive resistance if selected.",
        "Overturning check uses simplified stabilizing moment from vertical load at half footing dimension.",
        "Settlement is estimated using a simplified elastic settlement equation and should not replace detailed settlement analysis.",
        "Results require engineering judgment and should be reviewed against the project geotechnical report and applicable code requirements.",
    ]


def formula_sheet() -> List[Dict[str, str]]:
    return [
        {"name": "General bearing capacity", "formula": "q_ult = c Nc sc dc ic + q Nq sq dq iq + 0.5 gamma B Ngamma sgamma dgamma igamma"},
        {"name": "Surcharge", "formula": "q = gamma Df"},
        {"name": "Allowable gross bearing", "formula": "q_allow,gross = q_ult,gross / FS"},
        {"name": "Allowable net bearing", "formula": "q_allow,net = (q_ult,gross - q) / FS"},
        {"name": "Eccentricity", "formula": "e_x = |M_y| / P,  e_y = |M_x| / P"},
        {"name": "Effective area", "formula": "B' = B - 2e_x,  L' = L - 2e_y,  A' = B'L'"},
        {"name": "Effective applied bearing pressure", "formula": "q_eff = P / A'"},
        {"name": "Rigid footing contact pressure", "formula": "q = P/A ± M_x y/I_x ± M_y x/I_y"},
        {"name": "Rectangular footing inertia", "formula": "I_x = B L^3 / 12,  I_y = L B^3 / 12"},
        {"name": "Sliding FS", "formula": "FS_sliding = (P tan(delta) + c_a A + P_p) / H"},
        {"name": "Overturning FS", "formula": "FS_OT = M_resisting / M_overturning"},
        {"name": "Elastic settlement", "formula": "S = q_net B (1 - nu^2) I_s / E_s"},
    ]


def round_dict(d: Dict[str, float], ndigits: int = 4) -> Dict[str, float]:
    return {k: round(v, ndigits) for k, v in d.items()}


def round_nested(value: Any, ndigits: int = 4) -> Any:
    if isinstance(value, dict):
        return {k: round_nested(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [round_nested(v, ndigits) for v in value]
    if isinstance(value, float):
        return round(value, ndigits)
    return value
