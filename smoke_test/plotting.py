"""Plotting helpers for smoke test lightcurve products.

Strict policy (see AGENTS.md):
- Warnings are allowed (runner sets default filtering so receipts surface).
- NumPy FP errors raise immediately (divide/invalid/overflow/underflow).
- Transform assumptions (frames/units) documented inline; violations raise SmokeTestError.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import warnings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from astropy.coordinates import SkyCoord
import astropy.units as u

from .constants import REPO_ROOT
from .lightcurve_io import read_gulls_lightcurve
from .errors import SmokeTestError
from .utils import (
    derive_event_key,
    galactic_pm_to_icrs,
    compute_vbm_model,
)

# Enforce brittleness at import time too (in case this module is run outside the runner)
np.seterr(all="raise")

def _format_metric(value: float | None, precision: int = 3) -> str:
    # Deprecated: Avoid tolerant display of physics; prefer strict extraction below.
    if value is None or math.isnan(value):
        raise SmokeTestError("Unexpected missing metric in _format_metric; use strict extractors instead.")
    return f"{float(value):.{precision}f}"


## Header parsing moved to shared helper read_gulls_lightcurve


def _plot_photometry_only(
    lc_file: Path,
    output_dir: Path,
    title: str,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    true_flux: np.ndarray | None,
) -> Path:
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
    fig.tight_layout()
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Generated plot: {plot_file.name}")
    return plot_file


# Non-plotting helpers moved to utils.py (compute_vbm_model, galactic_pm_to_icrs, derive_event_key)


def _render_lensframe(
    lc_file: Path,
    output_dir: Path,
    time: np.ndarray,
    cmap: matplotlib.colors.Colormap,
    norm: Normalize,
    vbm_model: Dict[str, np.ndarray | str],
    true_x_vals: np.ndarray | None,
    true_y_vals: np.ndarray | None,
    meas_x: np.ndarray | None,
    meas_y: np.ndarray | None,
    src_x: np.ndarray | None,
    src_y: np.ndarray | None,
    pm_ref_alpha_float: float,
    pm_ref_delta_float: float,
    alpha_deg_float: float,
    true_E_mas: np.ndarray | None = None,
    true_N_mas: np.ndarray | None = None,
    bagle_shift_E_mas: np.ndarray | None = None,
    bagle_shift_N_mas: np.ndarray | None = None,
    bagle_thetaE_mas: float | None = None,
) -> Path:
    vbm_x = np.asarray(vbm_model["lens_x"])  # type: ignore[index]
    vbm_y = np.asarray(vbm_model["lens_y"])  # type: ignore[index]
    vbm_label = str(vbm_model["lens_label"])

    fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 6))

    if meas_x is not None and meas_y is not None:
        mask_meas = np.isfinite(meas_x) & np.isfinite(meas_y)
        if np.any(mask_meas):
            ax2.scatter(
                meas_x[mask_meas],
                meas_y[mask_meas],
                c=time[mask_meas],
                cmap=cmap,
                norm=norm,
                s=25,
                alpha=0.6,
                label="Blended centroid samples",
                zorder=1,
            )
    else:
        ax2.text(
            0.02,
            0.98,
            "Measured centroid not available",
            ha="left",
            va="top",
            transform=ax2.transAxes,
            fontsize=8,
        )

    if true_x_vals is not None and true_y_vals is not None:
        mask_true = np.isfinite(true_x_vals) & np.isfinite(true_y_vals)
        if np.any(mask_true):
            ax2.scatter(
                true_x_vals[mask_true],
                true_y_vals[mask_true],
                c=time[mask_true],
                cmap=cmap,
                norm=norm,
                s=20,
                marker="x",
                linewidths=0.9,
                alpha=1.0,
                label="True centroid",
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

    # Draw the blended centroid trajectory as a dashed cyan line on top of other artists
    if meas_x is not None and meas_y is not None:
        mask_meas = np.isfinite(meas_x) & np.isfinite(meas_y)
        if np.any(mask_meas):
            ax2.plot(
                np.asarray(meas_x)[mask_meas],
                np.asarray(meas_y)[mask_meas],
                linestyle="--",
                color="cyan",
                linewidth=2.0,
                alpha=0.95,
                label="Blended centroid trajectory",
                zorder=10,
            )
    ax2.set_xlabel("x_centroid (Einstein radii)")
    ax2.set_ylabel("y_centroid (Einstein radii)")
    ax2.set_title(f"Lens-frame Centroid: {lc_file.stem}")
    ax2.grid(True, alpha=0.3)

    segments_x = [vbm_x]
    segments_y = [vbm_y]
    if meas_x is not None and meas_y is not None:
        mask_meas = np.isfinite(meas_x) & np.isfinite(meas_y)
        if np.any(mask_meas):
            segments_x.append(np.asarray(meas_x)[mask_meas])
            segments_y.append(np.asarray(meas_y)[mask_meas])
    if true_x_vals is not None and true_y_vals is not None:
        mask_true = np.isfinite(true_x_vals) & np.isfinite(true_y_vals)
        if np.any(mask_true):
            segments_x.append(true_x_vals[mask_true])
            segments_y.append(true_y_vals[mask_true])
    all_x = np.concatenate(segments_x) if segments_x else np.array([0.0])
    all_y = np.concatenate(segments_y) if segments_y else np.array([0.0])
    x_min = float(np.nanmin(all_x))
    x_max = float(np.nanmax(all_x))
    y_min = float(np.nanmin(all_y))
    y_max = float(np.nanmax(all_y))
    x_c = 0.5 * (x_min + x_max)
    y_c = 0.5 * (y_min + y_max)
    half_span = max(x_max - x_min, y_max - y_min) * 0.5
    half_span = max(half_span, 1e-6)
    half_span *= 1.15
    ax2.set_xlim(x_c - half_span, x_c + half_span)
    ax2.set_ylim(y_c - half_span, y_c + half_span)
    ax2.set_aspect("equal", adjustable="box")
    

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar2 = fig2.colorbar(sm, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Time (days)")

    # --- Deterministic rotation mapping from sky (N,E) to lens-frame (x,y) using μ and α ---
    # From VBMicrolensing BinaryAstroLightCurve definitions (x=y1, y=y2):
    #   x =  u sinα − tn cosα
    #   y = −u cosα − tn sinα
    # With e_t along μ_rel and e_n = R(+90°) e_t, we have [tn,u]^T = R_NE2TN_U [N,E]^T where
    #   R_NE2TN_U = [[cos φ_μ, sin φ_μ], [−sin φ_μ, cos φ_μ]] and
    #   A(α) = [[−cosα, sinα], [−sinα, −cosα]].
    # Therefore R_NE→XY = A(α) @ R_NE2TN_U, which must be a proper rotation [[c,s], [−s,c]].
    if any(math.isnan(v) for v in (pm_ref_alpha_float, pm_ref_delta_float, alpha_deg_float)):
        raise SmokeTestError("Rotation diagnostic requires finite pm_ref_alpha, pm_ref_delta, and alpha_event.")
    muN = float(pm_ref_delta_float)
    muE = float(pm_ref_alpha_float)
    if muN == 0.0 and muE == 0.0:
        raise SmokeTestError("Relative proper motion vector is zero; cannot define along-track axis.")
    phi_mu = float(math.atan2(muE, muN))
    ct = float(math.cos(phi_mu)); st = float(math.sin(phi_mu))
    R_NE2TN_U = np.array([[ct, st], [-st, ct]], dtype=float)
    ca = float(math.cos(math.radians(alpha_deg_float))); sa = float(math.sin(math.radians(alpha_deg_float)))
    A_alpha = np.array([[-ca,  sa], [-sa, -ca]], dtype=float)
    R = A_alpha @ R_NE2TN_U
    c_est = float(R[0, 0]); s_est = float(R[0, 1])
    if not (np.allclose(R[1, 0], -s_est, atol=1e-6) and np.allclose(R[1, 1], c_est, atol=1e-6)):
        raise SmokeTestError("Derived rotation is not orthonormal within 1e-6; check pm_ref and alpha inputs.")

    # Temporary diagnostic: report rotation built from μ and α.
    # The rotation has φ_est such that [x, y]^T = R(φ_est) [N, E]^T with
    #   R(φ) = [[cos φ, sin φ], [-sin φ, cos φ]].
    phi_est_rad = float(np.arctan2(s_est, c_est))
    phi_mu_rad = float(math.atan2(muE, muN))
    alpha_rad = float(math.radians(alpha_deg_float))
    phi_axis_exp_rad = float(phi_mu_rad - alpha_rad)
    def _wrap_pi(a: float) -> float:
        return float((a + math.pi) % (2 * math.pi) - math.pi)
    def _wrap_pi_half(a: float) -> float:
        # map to (-pi/2, pi/2]
        return float(((a + 0.5 * math.pi) % math.pi) - 0.5 * math.pi)

    delta_mod_pi_rad = abs(_wrap_pi_half(phi_est_rad - phi_axis_exp_rad))
    delta_deg = abs(math.degrees(delta_mod_pi_rad))
    # Soft report: print and annotate in-figure (no warnings; env treats warnings as errors)
    # Report both candidates and which was chosen
    print(
        f"  NE↔lens rotation check for {lc_file.name}: "
        f"phi_est={math.degrees(phi_est_rad)%360:.2f}°, "
        f"phi_mu={math.degrees(phi_mu_rad)%360:.2f}°, "
        f"alpha={alpha_deg_float:.2f}°, "
        f"phi_mu−alpha={math.degrees(phi_axis_exp_rad)%360:.2f}° (|Δ| mod π={delta_deg:.2f}°)"
    )
    # Defer fail-fast until after we render overlays; collect message to raise later
    rotation_mismatch_msg: str | None = None
    if delta_mod_pi_rad > 1e-6:
        rotation_mismatch_msg = (
            f"Rotation mismatch: |phi_est - (phi_mu - alpha)| mod π = {math.degrees(delta_mod_pi_rad):.4f}° exceeds 1e-6 rad tolerance."
        )

    # Draw a faint expected-N arrow (from μ, α deterministic mapping) for visual sanity check
    n_exp = np.array([math.cos(phi_axis_exp_rad), -math.sin(phi_axis_exp_rad)], dtype=float)
    # Draw in the canonical VBM lens-frame basis (x=y1, y=y2) deterministically; no basis flips/swaps.
    # Use the same origin as the measured arrows (defined below) if available
    # Compute arrow geometry here as well to overlay expected-N just under the measured one
    span_tmp = 2 * half_span
    arrow_len_tmp = 0.18 * span_tmp
    padding_tmp = arrow_len_tmp + 0.03 * span_tmp
    x0_tmp = (x_c - half_span) + padding_tmp
    y0_tmp = (y_c + half_span) - padding_tmp
    ax2.annotate(
        "",
        xy=(x0_tmp + n_exp[0] * arrow_len_tmp, y0_tmp + n_exp[1] * arrow_len_tmp),
        xytext=(x0_tmp, y0_tmp),
        arrowprops=dict(arrowstyle="-", color="tab:blue", linewidth=1.0, alpha=0.35),
        zorder=11,
    )
    ax2.text(
        x0_tmp + n_exp[0] * arrow_len_tmp * 0.9,
        y0_tmp + n_exp[1] * arrow_len_tmp * 0.9,
        "N(sim)",
        color="tab:blue",
        fontsize=8,
        ha="center",
        va="center",
        alpha=0.6,
    )
    # Annotate the delta in a subtle way
    ax2.text(
        0.02,
        0.02,
        f"Δφ(N)≈{delta_deg:.1f}°",
        transform=ax2.transAxes,
        fontsize=8,
        color="dimgray",
        ha="left",
        va="bottom",
        alpha=0.8,
    )

    # If rotation was estimated, draw N and E arrows on lens-frame axes
    if c_est is not None and s_est is not None:
        # Lens-frame direction vectors for North and East (unit length in ER)
        n_dir = np.array([c_est, -s_est], dtype=float)
        e_dir = np.array([s_est,  c_est], dtype=float)
        # Normalize for safety
        def _safe_norm(v: np.ndarray) -> np.ndarray:
            n = float(np.hypot(v[0], v[1]))
            return v / n if n > 0 else v
        n_dir = _safe_norm(n_dir)
        e_dir = _safe_norm(e_dir)
        # Arrow placement near top-left, with origin set farther than one arrow length
        # from the bounds so tips cannot be clipped regardless of rotation.
        span = 2 * half_span
        arrow_len = 0.18 * span
        padding = arrow_len + 0.03 * span
        x0 = (x_c - half_span) + padding  # x_min + padding
        y0 = (y_c + half_span) - padding  # y_max - padding
        ax2.annotate("", xy=(x0 + e_dir[0]*arrow_len, y0 + e_dir[1]*arrow_len), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color="tab:green", linewidth=1.8), zorder=12)
        ax2.text(x0 + e_dir[0]*arrow_len*1.05, y0 + e_dir[1]*arrow_len*1.05, "E",
                 color="tab:green", fontsize=10, ha="center", va="center")
        ax2.annotate("", xy=(x0 + n_dir[0]*arrow_len, y0 + n_dir[1]*arrow_len), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", color="tab:blue", linewidth=1.8), zorder=12)
        ax2.text(x0 + n_dir[0]*arrow_len*1.05, y0 + n_dir[1]*arrow_len*1.05, "N",
                 color="tab:blue", fontsize=10, ha="center", va="center")

    # --- Source-frame panel (centroid relative to source position) ---
    if src_x is not None and src_y is not None:
        # Compute relative-to-source centroids (if available)
        rel_meas_x: np.ndarray | None = None
        rel_meas_y: np.ndarray | None = None
        rel_true_x: np.ndarray | None = None
        rel_true_y: np.ndarray | None = None

        if meas_x is not None and meas_y is not None:
            rel_meas_x = np.asarray(meas_x) - np.asarray(src_x)
            rel_meas_y = np.asarray(meas_y) - np.asarray(src_y)
        if true_x_vals is not None and true_y_vals is not None:
            rel_true_x = np.asarray(true_x_vals) - np.asarray(src_x)
            rel_true_y = np.asarray(true_y_vals) - np.asarray(src_y)

        if rel_meas_x is not None and rel_meas_y is not None:
            mask_m = np.isfinite(rel_meas_x) & np.isfinite(rel_meas_y)
            if np.any(mask_m):
                ax3.scatter(
                    rel_meas_x[mask_m],
                    rel_meas_y[mask_m],
                    c=time[mask_m],
                    cmap=cmap,
                    norm=norm,
                    s=25,
                    alpha=0.6,
                    label="Blended centroid samples (source)",
                    zorder=1,
                )
                # trajectory line
                ax3.plot(
                    rel_meas_x[mask_m],
                    rel_meas_y[mask_m],
                    linestyle="--",
                    color="cyan",
                    linewidth=2.0,
                    alpha=0.95,
                    label="Blended centroid trajectory (source)",
                    zorder=10,
                )
        if rel_true_x is not None and rel_true_y is not None:
            mask_t = np.isfinite(rel_true_x) & np.isfinite(rel_true_y)
            if np.any(mask_t):
                ax3.scatter(
                    rel_true_x[mask_t],
                    rel_true_y[mask_t],
                    c=time[mask_t],
                    cmap=cmap,
                    norm=norm,
                    s=18,
                    marker="x",
                    linewidths=0.9,
                    alpha=1.0,
                    label="True centroid (source)",
                    zorder=3,
                )

        # Optional overlay: BAGLE source-frame centroid trajectory mapped into lens-frame ER
        if (
            bagle_shift_E_mas is not None and bagle_shift_N_mas is not None and
            bagle_thetaE_mas is not None and np.isfinite(bagle_thetaE_mas) and bagle_thetaE_mas > 0
        ):
            e_er = np.asarray(bagle_shift_E_mas, dtype=float) / float(bagle_thetaE_mas)
            n_er = np.asarray(bagle_shift_N_mas, dtype=float) / float(bagle_thetaE_mas)
            mask_b = np.isfinite(e_er) & np.isfinite(n_er)
            if not np.any(mask_b):
                raise SmokeTestError("BAGLE overlay present but contains no finite samples.")
            # Apply previously-computed rotation [x, y]^T = R [N, E]^T; here we pass [n, e]
            x_b = c_est * n_er[mask_b] + s_est * e_er[mask_b]
            y_b = -s_est * n_er[mask_b] + c_est * e_er[mask_b]
            ax3.plot(
                x_b,
                y_b,
                linestyle=":",
                color="magenta",
                linewidth=1.8,
                alpha=0.9,
                label="BAGLE (source-frame)",
                zorder=11,
            )
            # Defer overlay consistency failure until after saving figure
            bagle_overlay_mismatch_msg: str | None = None
            if rel_true_x is not None and rel_true_y is not None:
                mask_t = np.isfinite(rel_true_x) & np.isfinite(rel_true_y)
                mask_common = np.zeros_like(mask_b, dtype=bool)
                mask_common[mask_b] = True
                mask_common &= mask_t
                if np.any(mask_common):
                    dx = x_b - rel_true_x[mask_common]
                    dy = y_b - rel_true_y[mask_common]
                    rmse = float(np.sqrt(np.mean(dx*dx + dy*dy)))
                    if not np.isfinite(rmse):
                        bagle_overlay_mismatch_msg = "BAGLE overlay RMSE is non-finite."
                    elif rmse > 5e-3:
                        bagle_overlay_mismatch_msg = (
                            f"BAGLE overlay mismatch in source-frame: RMSE={rmse:.4f} ER exceeds 0.005 ER tolerance."
                        )

        # Autoscale based on available data
        seg_x: list[np.ndarray] = []
        seg_y: list[np.ndarray] = []
        if rel_meas_x is not None and rel_meas_y is not None:
            mask_m = np.isfinite(rel_meas_x) & np.isfinite(rel_meas_y)
            if np.any(mask_m):
                seg_x.append(rel_meas_x[mask_m])
                seg_y.append(rel_meas_y[mask_m])
        if rel_true_x is not None and rel_true_y is not None:
            mask_t = np.isfinite(rel_true_x) & np.isfinite(rel_true_y)
            if np.any(mask_t):
                seg_x.append(rel_true_x[mask_t])
                seg_y.append(rel_true_y[mask_t])
        if seg_x and seg_y:
            allx = np.concatenate(seg_x)
            ally = np.concatenate(seg_y)
            xmin = float(np.nanmin(allx))
            xmax = float(np.nanmax(allx))
            ymin = float(np.nanmin(ally))
            ymax = float(np.nanmax(ally))
            xc = 0.5 * (xmin + xmax)
            yc = 0.5 * (ymin + ymax)
            half = max(xmax - xmin, ymax - ymin) * 0.5
            half = max(half, 1e-6) * 1.15
            ax3.set_xlim(xc - half, xc + half)
            ax3.set_ylim(yc - half, yc + half)
        ax3.set_aspect("equal", adjustable="box")
        ax3.set_xlabel("x_centroid - x_source (Einstein radii)")
        ax3.set_ylabel("y_centroid - y_source (Einstein radii)")
        ax3.set_title("Source-frame Centroid")
        ax3.grid(True, alpha=0.3)
        # Legends: place outside axes (above) to avoid covering data
        ax2.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=9)
        ax3.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False, fontsize=9)
        
        # Draw N/E arrows on the source-frame axes using the same rotation
        if c_est is not None and s_est is not None:
            n_dir = np.array([c_est, -s_est], dtype=float)
            e_dir = np.array([s_est,  c_est], dtype=float)
            def _safe_norm(v: np.ndarray) -> np.ndarray:
                n = float(np.hypot(v[0], v[1]))
                return v / n if n > 0 else v
            n_dir = _safe_norm(n_dir)
            e_dir = _safe_norm(e_dir)
            # Use current axis limits
            xlim = ax3.get_xlim(); ylim = ax3.get_ylim()
            span_x = float(xlim[1] - xlim[0])
            span_y = float(ylim[1] - ylim[0])
            span = max(span_x, span_y)
            arrow_len = 0.18 * span
            padding = arrow_len + 0.03 * span
            x0 = float(xlim[0] + padding)
            y0 = float(ylim[1] - padding)
            ax3.annotate("", xy=(x0 + e_dir[0]*arrow_len, y0 + e_dir[1]*arrow_len), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", color="tab:green", linewidth=1.8), zorder=12)
            ax3.text(x0 + e_dir[0]*arrow_len*1.05, y0 + e_dir[1]*arrow_len*1.05, "E",
                     color="tab:green", fontsize=10, ha="center", va="center")
            ax3.annotate("", xy=(x0 + n_dir[0]*arrow_len, y0 + n_dir[1]*arrow_len), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="->", color="tab:blue", linewidth=1.8), zorder=12)
            ax3.text(x0 + n_dir[0]*arrow_len*1.05, y0 + n_dir[1]*arrow_len*1.05, "N",
                     color="tab:blue", fontsize=10, ha="center", va="center")
    else:
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            "Source position columns not available",
            ha="center",
            va="center",
            transform=ax3.transAxes,
            fontsize=9,
        )

    lensframe_file = output_dir / f"{lc_file.stem}_lensframe_plot.png"
    # Reserve top margin for outside legends
    fig2.tight_layout(rect=[0, 0, 1, 0.9])
    fig2.savefig(lensframe_file, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    # After saving, raise any deferred mismatches to fail the run while keeping plots for forensics
    err_msgs: list[str] = []
    if 'rotation_mismatch_msg' in locals() and rotation_mismatch_msg:
        err_msgs.append(rotation_mismatch_msg)
    if 'bagle_overlay_mismatch_msg' in locals() and bagle_overlay_mismatch_msg:
        err_msgs.append(bagle_overlay_mismatch_msg)
    if err_msgs:
        raise SmokeTestError("; "+" ".join(err_msgs))
    return lensframe_file


def _render_astrometric_figure(
    lc_file: Path,
    output_dir: Path,
    title: str,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    true_flux: np.ndarray,
    true_N_mas: np.ndarray,
    true_E_mas: np.ndarray,
    meas_N_mas: np.ndarray,
    meas_E_mas: np.ndarray,
    meas_N_err_mas: np.ndarray,
    meas_E_err_mas: np.ndarray,
    true_ra_deg: np.ndarray,
    true_dec_deg: np.ndarray,
    meas_ra_deg: np.ndarray,
    meas_dec_deg: np.ndarray,
    meas_ra_err_deg: np.ndarray,
    meas_dec_err_deg: np.ndarray,
    vector_specs: List[Dict[str, float | str]],
    vbm_model: Dict[str, np.ndarray | str],
    true_x_vals: np.ndarray,
    true_y_vals: np.ndarray,
    meas_x: np.ndarray,
    meas_y: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    pm_ref_alpha_float: float,
    pm_ref_delta_float: float,
    alpha_deg_float: float,
    bagle_shift_E_mas: np.ndarray,
    bagle_shift_N_mas: np.ndarray,
    bagle_thetaE_mas: float,
) -> Tuple[Path, Path | None]:
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

    span_years: float | None = None
    if len(time):
        span_days = float(time.max() - time.min())
        if span_days > 0:
            span_years = span_days / 365.25

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

    # Overlay VBM model
    deg_to_mas = 3600.0 * 1000.0
    baseline_ra = true_ra_deg[0]
    baseline_dec = true_dec_deg[0]
    cos_dec0 = math.cos(math.radians(baseline_dec))
    if abs(cos_dec0) < 1e-6:
        cos_dec0 = 1e-6 if cos_dec0 >= 0 else -1e-6

    vbm_ra_offset_mas = np.asarray(vbm_model["sky_ra"], dtype=float)  # type: ignore[index]
    vbm_dec_offset_mas = np.asarray(vbm_model["sky_dec"], dtype=float)  # type: ignore[index]
    vbm_ra_abs = baseline_ra + (vbm_ra_offset_mas / (deg_to_mas * cos_dec0))
    vbm_dec_abs = baseline_dec + (vbm_dec_offset_mas / deg_to_mas)

    ax_radec.plot(
        vbm_ra_abs,
        vbm_dec_abs,
        color="tab:purple",
        linewidth=1.2,
        alpha=0.9,
        label=vbm_model.get("sky_label", "VBM BinaryAstroLightCurve (sky)"),
    )

    ax_radec.set_xlabel("RA (degrees)")
    ax_radec.set_ylabel("Dec (degrees)")
    ax_radec.set_title("Absolute Astrometric Position")
    ax_radec.grid(True, alpha=0.3)
    ax_radec.axis("equal")

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
            arrowprops=dict(color=spec["color"], arrowstyle="->", linewidth=1),
            zorder=5,
        )
        ax_radec.plot([], [], color=spec["color"], linewidth=2, label=spec["label"])
    
    ax_radec.legend(ncol=2, fontsize=8)

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

    # Overlay VBM model
    vbm_E_mas = np.asarray(vbm_model["sky_ra"], dtype=float)  # type: ignore[index]
    vbm_N_mas = np.asarray(vbm_model["sky_dec"], dtype=float)  # type: ignore[index]
    ax_ne.plot(
        vbm_E_mas,
        vbm_N_mas,
        color="tab:purple",
        linewidth=1.2,
        alpha=0.9,
        label=vbm_model.get("sky_label", "VBM BinaryAstroLightCurve (sky)"),
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
                arrowprops=dict(color=spec["color"], arrowstyle="->", linewidth=1),
                zorder=5,
            )
            ax_ne.plot([], [], color=spec["color"], linewidth=2, label=spec["label"])
    ax_ne.legend(ncol=2, fontsize=8)

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

    plot_file = output_dir / f"{lc_file.stem}_plot.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    lensframe_path: Path
    lensframe_path = _render_lensframe(
        lc_file,
        output_dir,
        time,
        cmap,
        norm,
        vbm_model,
        true_x_vals,
        true_y_vals,
        meas_x,
        meas_y,
        src_x,
        src_y,
        pm_ref_alpha_float=pm_ref_alpha_float,
        pm_ref_delta_float=pm_ref_delta_float,
        alpha_deg_float=alpha_deg_float,
        true_E_mas=true_E_mas,
        true_N_mas=true_N_mas,
        bagle_shift_E_mas=bagle_shift_E_mas,
        bagle_shift_N_mas=bagle_shift_N_mas,
        bagle_thetaE_mas=bagle_thetaE_mas,
    )

    return plot_file, lensframe_path


def plot_lightcurves(
    output_dir: Path,
    summaries: Dict[Tuple[int, int, int], Dict[str, float]],
    params: Dict[str, str],
) -> None:
    lc_files = sorted(output_dir.rglob("*.lc"))
    if not lc_files:
        return

    sim_zero_offset = 0.0
    sz = params.get("SIMULATION_ZERO_TIME")
    if sz:
        sim_zero_offset = float(sz) - 2450000.0

    astrometry_expected = False
    val = params.get("ASTROMETRY_ON")
    if val is not None:
        astrometry_expected = str(val).strip().lower() not in {"0", "false", "off"}

    validate_bagle_flag = False
    vb = params.get("VALIDATE_BAGLE")
    if vb is not None and str(vb).strip().lower() not in {"", "0", "false", "off", "none"}:
        validate_bagle_flag = True

    for lc_file in lc_files:
        df, meta = read_gulls_lightcurve(lc_file)
        if df.empty:
            continue
        planet_vals = meta.get('planet_vals')
        event_vals = meta.get('event_vals')

        column_names = list(df.columns)

        def _require_column(name: str) -> np.ndarray:
            if name not in column_names:
                raise SmokeTestError(f"Smoke test failed: column '{name}' missing in {lc_file.name}")
            return df[name].to_numpy(dtype=float, copy=False)

        def _optional_column(name: str) -> np.ndarray | None:
            if name not in column_names:
                return None
            return df[name].to_numpy(dtype=float, copy=False)

        event_key = derive_event_key(lc_file)
        summary = summaries.get(event_key)
        if summary is None:
            raise SmokeTestError(f"Smoke test failed: summary metrics missing for {lc_file.name}")

        title = f"Smoke Test: {lc_file.stem}"

        # Strict extractors for required values (fail-fast)
        def _require_summary(key: str, label: str, *, positive: bool = False) -> float:
            v = summary.get(key)
            if v is None or math.isnan(v):
                raise SmokeTestError(
                    f"Smoke test failed: missing or non-finite {label} for {lc_file.name}"
                )
            fv = float(v)
            if positive:
                if not (np.isfinite(fv) and fv > 0):
                    raise SmokeTestError(
                        f"Smoke test failed: {label} must be positive and finite for {lc_file.name}"
                    )
            return fv

        theta_e_float = _require_summary("theta_e", "theta_E")
        alpha_deg_float = _require_summary("alpha_event", "alpha_event")
        source_dist_float = _require_summary("source_dist", "source distance (pc)", positive=True)
        event_ra_float = _require_summary("event_ra", "event RA (deg)")
        event_dec_float = _require_summary("event_dec", "event Dec (deg)")

        # Also require geocentric relative PM now (used in subtitle and vectors)
        pm_ref_alpha_float = _require_summary("pm_ref_alpha", "pm_ref_alpha (mas/yr)")
        pm_ref_delta_float = _require_summary("pm_ref_delta", "pm_ref_delta (mas/yr)")

        # Build a strict subtitle using only required values
        subtitle = (
            f"theta_E={theta_e_float:.6f}, "
            f"Source D={source_dist_float:.3f} pc, "
            f"mu_ref=({pm_ref_alpha_float:.3f}, {pm_ref_delta_float:.3f}) mas/yr"
        )
        title = f"{title}\n{subtitle}"

        time = _require_column("Simulation_time")
        flux = _require_column("measured_relative_flux")
        flux_err = _require_column("measured_relative_flux_error")
        true_flux = _require_column("true_relative_flux")

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
        missing_astrom_cols = [col for col in astrom_cols if col not in column_names]
        has_astrom = not missing_astrom_cols
        if astrometry_expected and missing_astrom_cols:
            print(
                f"  Debug: {lc_file.name} missing astrometry columns: "
                + ", ".join(missing_astrom_cols)
            )
        if astrometry_expected and not has_astrom:
            raise SmokeTestError(
                f"Smoke test failed: astrometric columns missing in {lc_file.name}"
            )

        if not has_astrom:
            _plot_photometry_only(lc_file, output_dir, title, time, flux, flux_err, true_flux)
            continue

        true_N_mas = _require_column("true_N_centroid_mas")
        true_E_mas = _require_column("true_E_centroid_mas")
        meas_N_mas = _require_column("measured_N_centroid_mas")
        meas_E_mas = _require_column("measured_E_centroid_mas")
        meas_N_err_mas = _require_column("measured_N_centroid_error_mas")
        meas_E_err_mas = _require_column("measured_E_centroid_error_mas")
        true_ra_deg = _require_column("true_centroid_ra_deg")
        true_dec_deg = _require_column("true_centroid_dec_deg")
        meas_ra_deg = _require_column("measured_centroid_ra_deg")
        meas_dec_deg = _require_column("measured_centroid_dec_deg")
        meas_ra_err_deg = _require_column("measured_centroid_ra_error_deg")
        meas_dec_err_deg = _require_column("measured_centroid_dec_error_deg")
        true_x_vals = _require_column("true_x_centroid")
        true_y_vals = _require_column("true_y_centroid")
        meas_x = _require_column("x_centroid")
        meas_y = _require_column("y_centroid")
        src_x = _require_column("source_x")
        src_y = _require_column("source_y")

        # Optional: load BAGLE sidecar ONLY when --validate-bagle was used; if present it must be consistent.
        bagle_shift_E: np.ndarray | None = None
        bagle_shift_N: np.ndarray | None = None
        bagle_thetaE: float | None = None
        if validate_bagle_flag:
            sidecar_path = lc_file.parent / f"{lc_file.stem}_bagle.npz"
            if sidecar_path.exists():
                z = np.load(sidecar_path)
                required_keys = {"bagle_shift_E_mas", "bagle_shift_N_mas", "thetaE_mas"}
                missing = [k for k in required_keys if k not in z]
                if missing:
                    raise SmokeTestError(f"BAGLE sidecar {sidecar_path.name} missing keys: {', '.join(missing)}")
                # BAGLE explicitly returns centroid shifts as (East, North). The sidecar stores keys with that ordering.
                # Accept alias keys for backwards compatibility if present.
                if "bagle_shift_E_mas" in z and "bagle_shift_N_mas" in z:
                    bagle_shift_E = np.asarray(z["bagle_shift_E_mas"], dtype=float)
                    bagle_shift_N = np.asarray(z["bagle_shift_N_mas"], dtype=float)
                else:
                    # Fallback: accept older alias names 'shift_E'/'shift_N'
                    bagle_shift_E = np.asarray(z.get("shift_E", []), dtype=float)
                    bagle_shift_N = np.asarray(z.get("shift_N", []), dtype=float)
                bagle_thetaE = float(np.asarray(z["thetaE_mas"]).reshape(-1)[0])
                if not np.isfinite(bagle_thetaE) or bagle_thetaE <= 0:
                    raise SmokeTestError(f"BAGLE sidecar thetaE_mas must be positive and finite; got {bagle_thetaE}")

        # pm_ref_alpha_float / pm_ref_delta_float already required above

        # Strictly require source PM in Galactic coords, convert to ICRS
        src_vals = (
            summary.get("source_mul"),
            summary.get("source_mub"),
            summary.get("source_l"),
            summary.get("source_b"),
        )
        if not all(v is not None and not math.isnan(v) for v in src_vals):
            raise SmokeTestError(
                f"Smoke test failed: missing source proper motion (mul/mub) or l/b in summary for {lc_file.name}"
            )
        source_pm_icrs: Tuple[float, float] = galactic_pm_to_icrs(
            float(src_vals[2]),
            float(src_vals[3]),
            float(src_vals[0]),
            float(src_vals[1]),
        )

        # Lens PM vectors: if any lens PM fields are present, require all and be finite; else omit cleanly.
        lens_pm_icrs: Tuple[float, float] | None = None
        lens_vals = (
            summary.get("lens_mul"),
            summary.get("lens_mub"),
            summary.get("lens_l"),
            summary.get("lens_b"),
        )
        if any(v is not None for v in lens_vals):
            if not all(v is not None and not math.isnan(v) for v in lens_vals):
                raise SmokeTestError(
                    f"Smoke test failed: lens proper motion/l,b present but incomplete/non-finite for {lc_file.name}"
                )
            lens_pm_icrs = galactic_pm_to_icrs(
                float(lens_vals[2]),
                float(lens_vals[3]),
                float(lens_vals[0]),
                float(lens_vals[1]),
            )

        span_years = None
        if len(time):
            span_days = float(time.max() - time.min())
            if span_days > 0:
                span_years = span_days / 365.25

        vector_specs: List[Dict[str, float | str]] = []
        if span_years:
            if pm_ref_alpha_float is not None and pm_ref_delta_float is not None:
                vector_specs.append(
                    {
                        "label": "Relative proper motion (geocentric)",
                        "color": "black",
                        "pm_ra": pm_ref_alpha_float,
                        "pm_dec": pm_ref_delta_float,
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

        # Fail fast if required headers are missing
        if planet_vals is None:
            raise SmokeTestError(f"Smoke test failed: #Planet header missing for {lc_file.name}")
        if event_vals is None:
            raise SmokeTestError(f"Smoke test failed: #Event header missing for {lc_file.name}")

        vbm_model = compute_vbm_model(
            summary,
            planet_vals,
            event_vals,
            source_pm_icrs,
            lens_pm_icrs,
            float(theta_e_float) if theta_e_float is not None else float("nan"),
            float(source_dist_float) if source_dist_float is not None else float("nan"),
            float(event_ra_float) if event_ra_float is not None else float("nan"),
            float(event_dec_float) if event_dec_float is not None else float("nan"),
            float(alpha_deg_float),
            float(sim_zero_offset),
            time,
            true_x_vals,
            true_y_vals,
        )

        plot_file, lensframe_path = _render_astrometric_figure(
            lc_file,
            output_dir,
            title,
            time,
            flux,
            flux_err,
            true_flux,
            true_N_mas,
            true_E_mas,
            meas_N_mas,
            meas_E_mas,
            meas_N_err_mas,
            meas_E_err_mas,
            true_ra_deg,
            true_dec_deg,
            meas_ra_deg,
            meas_dec_deg,
            meas_ra_err_deg,
            meas_dec_err_deg,
            vector_specs,
            vbm_model,
            true_x_vals,
            true_y_vals,
            meas_x,
            meas_y,
            src_x,
            src_y,
            pm_ref_alpha_float,
            pm_ref_delta_float,
            alpha_deg_float,
            bagle_shift_E,
            bagle_shift_N,
            bagle_thetaE,
        )

        if astrometry_expected:
            if lensframe_path is None or not lensframe_path.exists():
                raise SmokeTestError(
                    f"Smoke test failed: missing lens-frame plot for {lc_file.name}"
                )
        print(f"  Generated plot: {plot_file.name}")
        if lensframe_path is not None:
            print(f"  Generated plot: {lensframe_path.name}")

__all__ = ["plot_lightcurves"]
