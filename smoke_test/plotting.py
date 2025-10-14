"""Plotting helpers for smoke test lightcurve products."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .constants import REPO_ROOT

try:
    import numpy as np
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

ASTROPY_AVAILABLE = False
if PLOTTING_AVAILABLE:
    try:
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        ASTROPY_AVAILABLE = True
    except ImportError:
        ASTROPY_AVAILABLE = False

VBM_AVAILABLE = False
VBM_CLASS = None
VBM_IMPORT_ERROR: str | None = None
if PLOTTING_AVAILABLE:
    try:
        import VBMicrolensing  # type: ignore
    except ImportError as exc:
        candidate = (REPO_ROOT.parent / "VBMicrolensing").resolve()
        if candidate.is_dir():
            sys.path.append(str(candidate))
            try:
                import VBMicrolensing  # type: ignore
            except ImportError as exc2:
                VBM_IMPORT_ERROR = str(exc2)
        else:
            VBM_IMPORT_ERROR = str(exc)
    else:
        VBM_CLASS = VBMicrolensing.VBMicrolensing  # type: ignore[attr-defined]
        VBM_AVAILABLE = True

VBM_PLOT_WARNING_EMITTED = False

def _derive_event_key(lc_file: Path) -> Tuple[int, int, int] | None:
    base = lc_file.stem.split(".", 1)[0]
    parts = base.rsplit("_", 3)
    if len(parts) < 4:
        return None
    try:
        return tuple(int(part) for part in parts[-3:])
    except ValueError:
        return None


def _format_metric(value: float | None, precision: int = 3) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def _galactic_pm_to_icrs(l_deg: float, b_deg: float, mu_l: float, mu_b: float) -> Tuple[float, float] | None:
    if not ASTROPY_AVAILABLE:
        return None
    try:
        coord = SkyCoord(
            l=l_deg * u.deg,
            b=b_deg * u.deg,
            pm_l_cosb=mu_l * u.mas / u.yr,
            pm_b=mu_b * u.mas / u.yr,
            frame="galactic",
        )
        icrs = coord.icrs
        return (
            icrs.pm_ra_cosdec.to_value(u.mas / u.yr),
            icrs.pm_dec.to_value(u.mas / u.yr),
        )
    except Exception:
        return None


def plot_lightcurves(
    output_dir: Path,
    summaries: Dict[Tuple[int, int, int], Dict[str, float]] | None = None,
    params: Dict[str, str] | None = None,
) -> None:
    if not PLOTTING_AVAILABLE:
        return

    lc_files = list(output_dir.rglob("*.lc"))
    if not lc_files:
        return

    global VBM_PLOT_WARNING_EMITTED

    sim_zero_offset = 0.0
    if params:
        sz = params.get("SIMULATION_ZERO_TIME")
        if sz:
            try:
                sim_zero_offset = float(sz) - 2450000.0
            except ValueError:
                sim_zero_offset = 0.0

    for lc_file in lc_files:
        try:
            planet_vals: List[float] | None = None
            event_vals: List[float] | None = None
            with lc_file.open(encoding="utf-8") as header_reader:
                for raw_header in header_reader:
                    if not raw_header.startswith("#"):
                        break
                    stripped = raw_header.strip()
                    if stripped.startswith("#Planet:"):
                        try:
                            planet_vals = [float(x) for x in stripped.split()[1:]]
                        except ValueError:
                            planet_vals = None
                    elif stripped.startswith("#Event:"):
                        try:
                            event_vals = [float(x) for x in stripped.split()[1:]]
                        except ValueError:
                            event_vals = None

            df = pd.read_csv(lc_file, sep=r"\s+", comment="#")
            if df.empty:
                continue

            summary = None
            if summaries:
                event_key = _derive_event_key(lc_file)
                if event_key is not None:
                    summary = summaries.get(event_key)

            title = f"Smoke Test: {lc_file.stem}"
            theta_e_float: float | None = None
            alpha_deg_float: float | None = None
            pi_n_float: float | None = None
            pi_e_float: float | None = None
            source_dist_float: float | None = None
            event_ra_float: float | None = None
            event_dec_float: float | None = None
            if summary:
                lens_mass = _format_metric(summary.get("lens_mass"))
                lens_dist = _format_metric(summary.get("lens_dist"))
                source_dist = _format_metric(summary.get("source_dist"))
                theta_e = _format_metric(summary.get("theta_e"))
                pm_alpha = _format_metric(summary.get("pm_alpha"))
                pm_delta = _format_metric(summary.get("pm_delta"))
                subtitle = (
                    f"Lens M={lens_mass} Msun, Lens D={lens_dist} pc, "
                    f"Source D={source_dist} pc, theta_E={theta_e}, "
                    f"mu_rel=({pm_alpha}, {pm_delta}) mas/yr"
                )
                title = f"{title}\n{subtitle}"
                val = summary.get("theta_e")
                if val is not None and not math.isnan(val):
                    theta_e_float = float(val)
                val = summary.get("alpha_event")
                if val is not None and not math.isnan(val):
                    alpha_deg_float = float(val)
                val = summary.get("pi_n")
                if val is not None and not math.isnan(val):
                    pi_n_float = float(val)
                val = summary.get("pi_e")
                if val is not None and not math.isnan(val):
                    pi_e_float = float(val)
                val = summary.get("source_dist")
                if val is not None and not math.isnan(val) and val > 0:
                    source_dist_float = float(val)
                val = summary.get("event_ra")
                if val is not None and not math.isnan(val):
                    event_ra_float = float(val)
                val = summary.get("event_dec")
                if val is not None and not math.isnan(val):
                    event_dec_float = float(val)
            if alpha_deg_float is None and event_vals and len(event_vals) >= 2:
                try:
                    alpha_deg_float = float(event_vals[1])
                except (ValueError, TypeError):
                    alpha_deg_float = None
            if alpha_deg_float is None:
                alpha_deg_float = 0.0

            time = df["Simulation_time"].values
            flux = df["measured_relative_flux"].values
            flux_err = df["measured_relative_flux_error"].values
            true_flux = df["true_relative_flux"].values if "true_relative_flux" in df.columns else None

            astrom_cols = [
                "true_N_centroid_mas",
                "true_E_centroid_mas",
                "measured_N_centroid_mas",
                "measured_E_centroid_mas",
                "measured_N_centroid_error_mas",
                "measured_E_centroid_error_mas",
                "true_centroid_ra_deg",
                "true_centroid_dec_deg",
                "measured_centroid_ra_deg",
                "measured_centroid_dec_deg",
                "measured_centroid_ra_error_deg",
                "measured_centroid_dec_error_deg",
            ]
            has_astrom = all(col in df.columns for col in astrom_cols)

            if has_astrom:
                true_N_mas = df["true_N_centroid_mas"].values
                true_E_mas = df["true_E_centroid_mas"].values
                meas_N_mas = df["measured_N_centroid_mas"].values
                meas_E_mas = df["measured_E_centroid_mas"].values
                meas_N_err_mas = df["measured_N_centroid_error_mas"].values
                meas_E_err_mas = df["measured_E_centroid_error_mas"].values
                true_ra_deg = df["true_centroid_ra_deg"].values
                true_dec_deg = df["true_centroid_dec_deg"].values
                meas_ra_deg = df["measured_centroid_ra_deg"].values
                meas_dec_deg = df["measured_centroid_dec_deg"].values
                meas_ra_err_deg = df["measured_centroid_ra_error_deg"].values
                meas_dec_err_deg = df["measured_centroid_dec_error_deg"].values
                true_x_vals = df["true_x_centroid"].values.astype(float, copy=False) if "true_x_centroid" in df.columns else None
                true_y_vals = df["true_y_centroid"].values.astype(float, copy=False) if "true_y_centroid" in df.columns else None
                pm_alpha_float = None
                pm_delta_float = None
                if summary:
                    pm_alpha_val = summary.get("pm_alpha")
                    pm_delta_val = summary.get("pm_delta")
                    if pm_alpha_val is not None and pm_delta_val is not None:
                        try:
                            pm_alpha_float = float(pm_alpha_val)
                            pm_delta_float = float(pm_delta_val)
                            if math.isnan(pm_alpha_float) or math.isnan(pm_delta_float):
                                pm_alpha_float = None
                                pm_delta_float = None
                        except (TypeError, ValueError):
                            pm_alpha_float = None
                            pm_delta_float = None

                source_pm_icrs: Tuple[float, float] | None = None
                lens_pm_icrs: Tuple[float, float] | None = None
                if summary and ASTROPY_AVAILABLE:
                    src_vals = (
                        summary.get("source_mul"),
                        summary.get("source_mub"),
                        summary.get("source_l"),
                        summary.get("source_b"),
                    )
                    if all(v is not None and not math.isnan(v) for v in src_vals):
                        source_pm_icrs = _galactic_pm_to_icrs(
                            float(src_vals[2]),
                            float(src_vals[3]),
                            float(src_vals[0]),
                            float(src_vals[1]),
                        )
                    lens_vals = (
                        summary.get("lens_mul"),
                        summary.get("lens_mub"),
                        summary.get("lens_l"),
                        summary.get("lens_b"),
                    )
                    if all(v is not None and not math.isnan(v) for v in lens_vals):
                        lens_pm_icrs = _galactic_pm_to_icrs(
                            float(lens_vals[2]),
                            float(lens_vals[3]),
                            float(lens_vals[0]),
                            float(lens_vals[1]),
                        )

            if has_astrom:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(title, fontsize=14)
                ax_light = axes[0, 0]
                ax_time = axes[0, 1]
                ax_radec = axes[1, 0]
                ax_ne = axes[1, 1]

                ax_light.errorbar(
                    time,
                    flux,
                    yerr=flux_err,
                    fmt="o",
                    markersize=2,
                    alpha=0.5,
                    color="C0",
                    label="Measured",
                    zorder=1,
                )
                if true_flux is not None:
                    ax_light.plot(
                        time,
                        true_flux,
                        "-",
                        linewidth=1.5,
                        color="red",
                        label="True",
                        zorder=2,
                        alpha=0.8,
                    )
                ax_light.axhline(
                    1.0,
                    color="k",
                    linestyle="--",
                    linewidth=1.5,
                    label="Baseline",
                    zorder=3,
                )
                ax_light.set_xlabel("Time (days)")
                ax_light.set_ylabel("Relative Flux")
                ax_light.set_title("Light Curve")
                ax_light.legend()
                ax_light.grid(True, alpha=0.3)

                norm = Normalize(vmin=np.min(time), vmax=np.max(time)) if len(time) else Normalize(0, 1)
                cmap = plt.get_cmap("plasma")
                span_days = float(time.max() - time.min()) if len(time) else 0.0
                span_years = span_days / 365.25 if span_days > 0 else None

                vector_specs: List[Dict[str, float | str]] = []
                if span_years:
                    if summary and pm_alpha_float is not None and pm_delta_float is not None:
                        vector_specs.append(
                            {
                                "label": "Relative proper motion (heliocentric)",
                                "color": "black",
                                "pm_ra": pm_alpha_float,
                                "pm_dec": pm_delta_float,
                            }
                        )
                    if source_pm_icrs:
                        vector_specs.append(
                            {
                                "label": "Source proper motion (heliocentric)",
                                "color": "tab:blue",
                                "pm_ra": source_pm_icrs[0],
                                "pm_dec": source_pm_icrs[1],
                            }
                        )
                    if lens_pm_icrs:
                        vector_specs.append(
                            {
                                "label": "Lens proper motion (heliocentric)",
                                "color": "tab:red",
                                "pm_ra": lens_pm_icrs[0],
                                "pm_dec": lens_pm_icrs[1],
                            }
                        )

                vbm_model: Tuple[np.ndarray, np.ndarray, str] | None = None
                if VBM_AVAILABLE and summary:
                    if not ASTROPY_AVAILABLE:
                        if not VBM_PLOT_WARNING_EMITTED:
                            print("  Warning: astropy not available; skipping VBM centroid overlay.")
                            VBM_PLOT_WARNING_EMITTED = True
                    elif (
                        planet_vals
                        and len(planet_vals) >= 6
                        and event_vals
                        and len(event_vals) >= 8
                    ):
                        try:
                            q_val = float(planet_vals[4])
                            s_val = float(planet_vals[5])
                            if q_val > 0 and s_val > 0:
                                rho_val = summary.get("rho")
                                if rho_val is None or math.isnan(rho_val):
                                    rho_val = float(event_vals[7])
                                tE_val = summary.get("tE_ref")
                                if tE_val is None or math.isnan(tE_val):
                                    tE_val = float(event_vals[6])
                                u0_val = summary.get("u0")
                                if u0_val is None or math.isnan(u0_val):
                                    u0_val = float(event_vals[0])
                                alpha_deg = summary.get("alpha_event")
                                if alpha_deg is None or math.isnan(alpha_deg):
                                    alpha_deg = float(event_vals[1])
                                t0_val = summary.get("t0")
                                if t0_val is None or math.isnan(t0_val):
                                    t0_val = float(event_vals[2])
                                pi_n_val = summary.get("pi_n")
                                pi_e_val = summary.get("pi_e")
                                if pi_n_val is None or math.isnan(pi_n_val):
                                    pi_n_val = 0.0
                                if pi_e_val is None or math.isnan(pi_e_val):
                                    pi_e_val = 0.0
                                event_theta_e = theta_e_float
                                if (
                                    event_theta_e is not None
                                    and event_theta_e > 0
                                    and source_dist_float
                                    and source_dist_float > 0
                                    and source_pm_icrs
                                    and event_ra_float is not None
                                    and event_dec_float is not None
                                    and alpha_deg_float is not None
                                ):
                                    pi_s_val = 1.0 / source_dist_float
                                    if (
                                        pi_s_val > 0
                                        and rho_val is not None
                                        and float(rho_val) > 0
                                        and tE_val is not None
                                        and float(tE_val) > 0
                                    ):
                                        vbm = VBM_CLASS()  # type: ignore[operator]
                                        coord_str = SkyCoord(
                                            ra=float(event_ra_float) * u.deg,
                                            dec=float(event_dec_float) * u.deg,
                                        ).to_string("hmsdms")
                                        params_vbm = [
                                            math.log(float(s_val)),
                                            math.log(float(q_val)),
                                            float(u0_val),
                                            math.radians(float(alpha_deg)),
                                            math.log(float(rho_val)),
                                            math.log(float(tE_val)),
                                            float(t0_val) + sim_zero_offset,
                                            float(pi_n_val),
                                            float(pi_e_val),
                                            float(source_pm_icrs[1]),
                                            float(source_pm_icrs[0]),
                                            float(pi_s_val),
                                            float(event_theta_e),
                                        ]
                                        times_vbm = df["Simulation_time"].values + sim_zero_offset
                                        results = vbm.BinaryAstroLightCurve(params_vbm, times_vbm)
                                        y1 = np.array(results[5], dtype=float)
                                        y2 = np.array(results[6], dtype=float)
                                        vbm_label = "VBM BinaryAstroLightCurve"
                                        if (
                                            true_x_vals is not None
                                            and true_y_vals is not None
                                            and len(true_x_vals) == len(y1)
                                        ):
                                            combos = [
                                                ("x= y1, y= y2", y1, y2),
                                                ("x=-y1, y= y2", -y1, y2),
                                                ("x= y1, y=-y2", y1, -y2),
                                                ("x=-y1, y=-y2", -y1, -y2),
                                                ("x= y2, y= y1", y2, y1),
                                                ("x=-y2, y= y1", -y2, y1),
                                                ("x= y2, y=-y1", y2, -y1),
                                                ("x=-y2, y=-y1", -y2, -y1),
                                            ]
                                            best = None
                                            best_err = None
                                            for label, cand_x, cand_y in combos:
                                                err = np.nanmean((cand_x - true_x_vals) ** 2 + (cand_y - true_y_vals) ** 2)
                                                if best_err is None or err < best_err:
                                                    best_err = err
                                                    best = (label, cand_x, cand_y)
                                            if best:
                                                vbm_label = f"VBM BinaryAstroLightCurve ({best[0]}, no parallax)"
                                                vbm_x, vbm_y = best[1], best[2]
                                            else:
                                                vbm_x, vbm_y = y1, y2
                                        else:
                                            vbm_x, vbm_y = y1, y2
                                        if len(vbm_x) == len(times_vbm):
                                            vbm_model = (vbm_x, vbm_y, vbm_label)
                        except Exception as err:
                            if not VBM_PLOT_WARNING_EMITTED:
                                print(f"  Warning: VBM centroid reconstruction failed for {lc_file.name}: {err}")
                                VBM_PLOT_WARNING_EMITTED = True

                ax_radec.errorbar(
                    meas_ra_deg,
                    meas_dec_deg,
                    xerr=meas_ra_err_deg,
                    yerr=meas_dec_err_deg,
                    fmt="none",
                    ecolor="lightgray",
                    alpha=0.5,
                    capsize=2,
                    zorder=0,
                )
                sc_ra = ax_radec.scatter(
                    meas_ra_deg,
                    meas_dec_deg,
                    c=time,
                    cmap=cmap,
                    norm=norm,
                    s=25,
                    alpha=0.5,
                    label="Measured",
                    zorder=1,
                )
                ax_radec.plot(
                    true_ra_deg,
                    true_dec_deg,
                    color="black",
                    linewidth=1.2,
                    alpha=0.8,
                    label="True track",
                    zorder=4,
                )
                ax_radec.scatter(
                    true_ra_deg,
                    true_dec_deg,
                    c=time,
                    cmap=cmap,
                    norm=norm,
                    s=18,
                    marker="x",
                    linewidths=0.8,
                    alpha=1.0,
                    label="True samples",
                    zorder=3,
                )
                ax_radec.set_xlabel("RA (degrees)")
                ax_radec.set_ylabel("Dec (degrees)")
                ax_radec.set_title("Absolute Astrometric Position")
                ax_radec.grid(True, alpha=0.3)
                ax_radec.axis("equal")

                if span_years and vector_specs:
                    start_ra = true_ra_deg[0]
                    start_dec = true_dec_deg[0]
                    cos_dec = math.cos(math.radians(start_dec))
                    if abs(cos_dec) < 1e-6:
                        cos_dec = 1e-6 if cos_dec >= 0 else -1e-6
                    for spec in vector_specs:
                        pm_ra = spec["pm_ra"]
                        pm_dec = spec["pm_dec"]
                        delta_ra_deg = (pm_ra * span_years) / (3600000.0 * cos_dec)
                        delta_dec_deg = (pm_dec * span_years) / 3600000.0
                        end_ra = start_ra + delta_ra_deg
                        end_dec = start_dec + delta_dec_deg
                        ax_radec.annotate(
                            "",
                            xy=(end_ra, end_dec),
                            xytext=(start_ra, start_dec),
                            arrowprops=dict(color=spec["color"], arrowstyle="-|>", linewidth=2),
                            zorder=5,
                        )
                        ax_radec.plot([], [], color=spec["color"], linewidth=2, label=spec["label"])
                ax_radec.legend()

                ax_ne.plot(
                    true_E_mas,
                    true_N_mas,
                    color="black",
                    linewidth=1.2,
                    alpha=0.8,
                    label="True track",
                    zorder=4,
                )
                ax_ne.scatter(
                    true_E_mas,
                    true_N_mas,
                    c=time,
                    cmap=cmap,
                    norm=norm,
                    s=18,
                    marker="x",
                    linewidths=0.8,
                    alpha=1.0,
                    label="True samples",
                    zorder=3,
                )
                ax_ne.errorbar(
                    meas_E_mas,
                    meas_N_mas,
                    xerr=meas_E_err_mas,
                    yerr=meas_N_err_mas,
                    fmt="none",
                    ecolor="lightgray",
                    alpha=0.5,
                    capsize=2,
                    zorder=0,
                )
                ax_ne.scatter(
                    meas_E_mas,
                    meas_N_mas,
                    c=time,
                    cmap=cmap,
                    norm=norm,
                    s=25,
                    alpha=0.5,
                    label="Measured",
                    zorder=1,
                )
                ax_ne.set_xlabel("ΔEast (mas)")
                ax_ne.set_ylabel("ΔNorth (mas)")
                ax_ne.set_title("Astrometric Centroid (N/E), Relative to the Lens")
                ax_ne.grid(True, alpha=0.3)
                ax_ne.axis("equal")

                if span_years and vector_specs:
                    start_E = true_E_mas[0]
                    start_N = true_N_mas[0]
                    for spec in vector_specs:
                        pm_ra = spec["pm_ra"]
                        pm_dec = spec["pm_dec"]
                        delta_E_mas = pm_ra * span_years
                        delta_N_mas = pm_dec * span_years
                        end_E = start_E + delta_E_mas
                        end_N = start_N + delta_N_mas
                        ax_ne.annotate(
                            "",
                            xy=(end_E, end_N),
                            xytext=(start_E, start_N),
                            arrowprops=dict(color=spec["color"], arrowstyle="-|>", linewidth=2),
                            zorder=5,
                        )
                        ax_ne.plot([], [], color=spec["color"], linewidth=2, label=spec["label"])
                ax_ne.legend()

                fig.tight_layout(rect=[0, 0.12, 1, 1])
                cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.025])
                cbar = fig.colorbar(sc_ra, cax=cbar_ax, orientation="horizontal")
                cbar.set_label("Time (days)")

                ax_time.plot(time, true_N_mas, "b-", label="True N", linewidth=2, alpha=0.7)
                ax_time.plot(time, meas_N_mas, "r.", label="Meas N", markersize=2, alpha=0.5)
                ax_time.plot(time, true_E_mas, "g-", label="True E", linewidth=2, alpha=0.7)
                ax_time.plot(time, meas_E_mas, "m.", label="Meas E", markersize=2, alpha=0.5)
                ax_time.set_xlabel("Time (days)")
                ax_time.set_ylabel("Centroid Shift (mas)")
                ax_time.set_title("Astrometric Timeseries (N and E)")
                ax_time.legend(fontsize=8)
                ax_time.grid(True, alpha=0.3)
            else:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                fig.suptitle(title, fontsize=14)
                ax.errorbar(
                    time,
                    flux,
                    yerr=flux_err,
                    fmt="o",
                    markersize=2,
                    alpha=0.5,
                    color="C0",
                    label="Measured",
                    zorder=1,
                )
                if true_flux is not None:
                    ax.plot(
                        time,
                        true_flux,
                        "-",
                        linewidth=1.5,
                        color="red",
                        label="True",
                        zorder=2,
                        alpha=0.8,
                    )
                ax.axhline(1.0, color="k", linestyle="--", linewidth=1.5, label="Baseline", zorder=3)
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Relative Flux")
                ax.set_title("Light Curve")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plot_file = output_dir / f"{lc_file.stem}_plot.png"
            if not has_astrom:
                fig.tight_layout()
            fig.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close(fig)

            if has_astrom and vbm_model is not None and "x_centroid" in df.columns and "y_centroid" in df.columns:
                vbm_x, vbm_y, vbm_label = vbm_model
                meas_x = df["x_centroid"].values.astype(float, copy=False)
                meas_y = df["y_centroid"].values.astype(float, copy=False)
                true_x = true_x_vals
                true_y = true_y_vals
                mask_meas = np.isfinite(meas_x) & np.isfinite(meas_y)
                fig2, ax2 = plt.subplots(figsize=(6, 6))
                sc2 = ax2.scatter(
                    meas_x[mask_meas],
                    meas_y[mask_meas],
                    c=time[mask_meas],
                    cmap=cmap,
                    norm=norm,
                    s=25,
                    alpha=0.6,
                    label="Measured centroid",
                    zorder=1,
                )
                if true_x is not None and true_y is not None:
                    mask_true = np.isfinite(true_x) & np.isfinite(true_y)
                    ax2.plot(
                        true_x[mask_true],
                        true_y[mask_true],
                        linestyle="--",
                        linewidth=1.2,
                        color="gray",
                        label="Stored true centroid",
                        zorder=2,
                    )
                ax2.plot(
                    vbm_x,
                    vbm_y,
                    color="black",
                    linewidth=1.5,
                    label=vbm_label,
                    zorder=3,
                )
                ax2.set_xlabel("x_centroid (Einstein radii)")
                ax2.set_ylabel("y_centroid (Einstein radii)")
                ax2.set_title(f"Lens-frame Centroid: {lc_file.stem}")
                ax2.grid(True, alpha=0.3)
                all_x_segments = [vbm_x]
                all_y_segments = [vbm_y]
                if np.any(mask_meas):
                    all_x_segments.append(meas_x[mask_meas])
                    all_y_segments.append(meas_y[mask_meas])
                if true_x is not None and true_y is not None:
                    mask_true = np.isfinite(true_x) & np.isfinite(true_y)
                    if np.any(mask_true):
                        all_x_segments.append(true_x[mask_true])
                        all_y_segments.append(true_y[mask_true])
                all_x = np.concatenate(all_x_segments) if all_x_segments else np.array([0.0])
                all_y = np.concatenate(all_y_segments) if all_y_segments else np.array([0.0])
                x_min = float(np.nanmin(all_x))
                x_max = float(np.nanmax(all_x))
                y_min = float(np.nanmin(all_y))
                y_max = float(np.nanmax(all_y))
                x_c = 0.5 * (x_min + x_max)
                y_c = 0.5 * (y_min + y_max)
                half_span = max(x_max - x_min, y_max - y_min) * 0.5
                half_span = max(half_span, 1e-6)
                margin = half_span * 0.15
                half_span += margin
                ax2.set_xlim(x_c - half_span, x_c + half_span)
                ax2.set_ylim(y_c - half_span, y_c + half_span)
                ax2.set_aspect("equal", adjustable="box")
                ax2.legend(loc="upper left")
                cbar2 = fig2.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
                cbar2.set_label("Time (days)")
                lensframe_file = output_dir / f"{lc_file.stem}_lensframe_plot.png"
                fig2.tight_layout()
                fig2.savefig(lensframe_file, dpi=150, bbox_inches="tight")
                plt.close(fig2)
                print(f"  Generated plot: {lensframe_file.name}")

            print(f"  Generated plot: {plot_file.name}")
        except Exception as exc:
            print(f"  Warning: Could not plot {lc_file.name}: {exc}")
            continue


__all__ = ["plot_lightcurves", "PLOTTING_AVAILABLE", "VBM_AVAILABLE", "VBM_IMPORT_ERROR"]
