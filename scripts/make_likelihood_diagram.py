"""Diagram of the image-position likelihood: where each position lives and how the covariance is built.

A toy lens, not a fitted cluster.  One source is lensed by a parametric halo plus one realization of the Gaussian
random field; its images are displaced by one draw of the image-plane Gaussian process and of the measurement error.
The error scales are larger than in a real catalog so the offsets can be seen.  Every contour is a unit-Mahalanobis
ellipse of the covariance named in the legend.

Run with the lenscluster interpreter.  Writes paper/figures/position_likelihood_diagram.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root

from kappaccino.plotting.style import CRITICAL_COLOR, FIGURE_STYLE, GREEN, ORANGE, PINK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "position_likelihood_diagram.pdf"
STYLE = ROOT / "src" / "kappaccino" / "plotting" / "style.mplstyle"
INK, MUTED = "#22262B", "#6B7178"
MEAS, GP, GRF = ORANGE, GREEN, PINK   # one colour per covariance term, in every panel

# Toy lens: singular isothermal ellipsoid, Einstein radius B (arcsec) and axis ratio Q, major axis along x.
B, Q = 20.0, 0.62
# Error scales (arcsec), exaggerated for legibility.
SIGMA_GRF, LAMBDA_B, N_KAPPA = 0.12, 12.0, 4.0
SIGMA_GP, ELL_GP = 0.08, 4.0
SIGMA_MEAS = 0.04
FOLD_DEPTH = 0.05       # distance of the source inside the caustic
SEED_FIELD, SEED_ERRORS = 1, 377


def parametric(xy):
    """Deflection and Hessian of the potential of the toy halo at points (..., 2) (Kormann et al. 1994)."""
    x, y = xy[..., 1], xy[..., 0]                       # the formulas put the major axis along their second coordinate
    x = np.where((x == 0) & (y == 0), 1e-9, x)
    root_q, qp = np.sqrt(Q), np.sqrt(1 - Q**2)
    phi = np.arctan2(y, x)
    kappa = B * root_q / (2 * np.sqrt(x**2 + Q**2 * y**2))
    a1 = B * root_q / qp * np.arcsinh(qp * np.cos(phi) / Q)
    a2 = B * root_q / qp * np.arcsin(qp * np.sin(phi))
    h11, h22, h12 = 2 * kappa * np.sin(phi)**2, 2 * kappa * np.cos(phi)**2, -2 * kappa * np.sin(phi) * np.cos(phi)
    return np.stack([a2, a1], -1), np.stack([np.stack([h22, h12], -1), np.stack([h12, h11], -1)], -2)


class RandomField:
    """One realization of the deflection field with the broken power-law convergence spectrum, and its covariance."""

    def __init__(self, seed, n=512, size=160.0):
        step = size / n
        k1 = 2 * np.pi * np.fft.fftfreq(n, step)
        kx, ky = np.meshgrid(k1, k1)
        k = np.hypot(kx, ky)
        k[0, 0] = 1.0
        power = 1 / (1 + (k / (2 * np.pi / LAMBDA_B))**N_KAPPA)
        power[(k < 2 * np.pi / 50) | (k > 2 * np.pi / 1.0)] = 0
        power[0, 0] = 0
        spectra = {"xx": 4 * kx**2 / k**4 * power, "yy": 4 * ky**2 / k**4 * power, "xy": 4 * kx * ky / k**4 * power}
        variance = .5 * (np.fft.ifft2(spectra["xx"]).real[0, 0] + np.fft.ifft2(spectra["yy"]).real[0, 0])
        scale = SIGMA_GRF**2 / variance
        white = np.fft.fft2(np.random.default_rng(seed).standard_normal((n, n)))
        kappa_hat = np.sqrt(scale * power) * white
        psi_hat = -2 * kappa_hat / k**2
        grid = (np.arange(n) - n // 2) * step
        spline = lambda field: RectBivariateSpline(grid, grid, np.fft.fftshift(field).T, kx=3, ky=3)
        self.grid = grid
        self.kappa = np.fft.fftshift(np.fft.ifft2(kappa_hat).real)
        self.ax = spline(np.fft.ifft2(1j * kx * psi_hat).real)
        self.ay = spline(np.fft.ifft2(1j * ky * psi_hat).real)
        self.kernel = {key: spline(scale * np.fft.ifft2(value).real) for key, value in spectra.items()}

    def deflection(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        return np.stack([self.ax.ev(x, y), self.ay.ev(x, y)], -1)

    def hessian(self, xy):
        x, y = xy[..., 0], xy[..., 1]
        return np.array([[self.ax.ev(x, y, dx=1), self.ax.ev(x, y, dy=1)],
                         [self.ay.ev(x, y, dx=1), self.ay.ev(x, y, dy=1)]])

    def covariance(self, r):
        """2 x 2 covariance of the deflections at two points separated by r."""
        kxx, kyy, kxy = (float(self.kernel[key].ev(r[0], r[1])) for key in ("xx", "yy", "xy"))
        return np.array([[kxx, kxy], [kxy, kyy]])


def jacobian(xy):
    return np.eye(2) - parametric(xy)[1]


def back_project(xy):
    """Source-plane position of an image-plane position in the parametric lens."""
    return xy - parametric(xy)[0]


def critical_and_caustic():
    """Tangential critical curve of the parametric lens and its caustic, as closed polylines."""
    axis = np.linspace(-40, 40, 1201)
    xx, yy = np.meshgrid(axis, axis)
    points = np.stack([xx, yy], -1)
    a = np.eye(2) - parametric(points)[1]
    det = a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0]
    figure, ax = plt.subplots()
    segments = ax.contour(xx, yy, det, levels=[0]).allsegs[0]
    plt.close(figure)
    critical = max(segments, key=len)
    return critical, back_project(critical)


def find_images(source, field):
    """All images of a source in the full lens (parametric halo plus the field), outside the central 2 arcsec."""
    def equation(theta):
        return theta - parametric(theta)[0] - field.deflection(theta) - source

    def derivative(theta):
        return jacobian(theta) - field.hessian(theta)

    found = []
    for radius in np.linspace(6, 34, 9):
        for angle in np.linspace(0, 2 * np.pi, 36, endpoint=False):
            solution = root(equation, radius * np.array([np.cos(angle), np.sin(angle)]), jac=derivative, tol=1e-12)
            if not solution.success or np.hypot(*solution.x) < 2 or np.abs(equation(solution.x)).max() > 1e-8:
                continue
            if all(np.hypot(*(solution.x - other)) > 1e-3 for other in found):
                found.append(solution.x)
    return np.array(found)


def ellipse(covariance, **style):
    values, vectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    return Ellipse((0, 0), 2 * np.sqrt(max(values[1], 0)), 2 * np.sqrt(max(values[0], 0)), angle=angle, fill=False, **style)


def arrow(ax, start, end, color, width=1.3, head=7, **style):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head, color=color, lw=width,
                                 shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt", **style))


def lensed_family(field_seed=SEED_FIELD):
    """The field, the curves of the parametric lens, the source and where the full lens puts its images."""
    field = RandomField(field_seed)
    critical, caustic = critical_and_caustic()
    # Source just inside a fold of the caustic, about 40 degrees from the x axis.
    angles = np.degrees(np.arctan2(caustic[:, 1], caustic[:, 0]))
    fold = caustic[np.argmin(np.abs(angles - 38))]
    source = fold * (1 - FOLD_DEPTH / np.hypot(*fold))
    lens = find_images(source, field)
    lens = lens[np.argsort(-np.arctan2(lens[:, 1], lens[:, 0]))]
    return dict(field=field, critical=critical, caustic=caustic, source=source, lens=lens)


def build(error_seed=SEED_ERRORS, family=None):
    """Positions, Jacobians and covariances of the toy family."""
    family = family or lensed_family()
    field, lens = family["field"], family["lens"]
    n = len(lens)
    rng = np.random.default_rng(error_seed)
    separation = lens[:, None] - lens[None]
    gp_kernel = np.exp(-(separation**2).sum(-1) / (2 * ELL_GP**2))
    chol = np.linalg.cholesky(SIGMA_GP**2 * gp_kernel + 1e-12 * np.eye(n))
    displacement = chol @ rng.standard_normal((n, 2))
    true = lens + displacement
    measured = true + SIGMA_MEAS * rng.standard_normal((n, 2))
    jac = np.array([jacobian(point) for point in measured])
    c_meas, c_gp, c_grf = (np.zeros((n, 2, n, 2)) for _ in range(3))
    for i in range(n):
        c_meas[i, :, i, :] = SIGMA_MEAS**2 * jac[i] @ jac[i].T
        for j in range(n):
            r = measured[i] - measured[j]
            c_gp[i, :, j, :] = SIGMA_GP**2 * np.exp(-(r**2).sum() / (2 * ELL_GP**2)) * jac[i] @ jac[j].T
            c_grf[i, :, j, :] = field.covariance(r)
    highlight = int(np.argmax([abs(np.linalg.det(a)) for a in jac]))
    near = int(np.argmin([abs(np.linalg.det(a)) for a in jac]))
    return dict(family, highlight=highlight, near=near, true=true, measured=measured, jac=jac, c_meas=c_meas, c_gp=c_gp,
                c_grf=c_grf)


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def away(points, target, distance):
    """Offset (points) that moves a label at `target` away from the centroid of `points`."""
    direction = np.asarray(target) - np.mean(points, 0)
    norm = np.hypot(*direction)
    return distance * (direction / norm if norm > 0 else np.array([0, 1.0]))


def beside(start, end, points, distance):
    """Offset (points) perpendicular to the segment, on the side away from the centroid of `points`."""
    along = np.asarray(end) - np.asarray(start)
    normal = np.array([-along[1], along[0]]) / np.hypot(*along)
    middle = .5 * (np.asarray(start) + np.asarray(end))
    return distance * normal * (1 if normal @ (middle - np.mean(points, 0)) >= 0 else -1)


def label(ax, text, anchor, offset, color=INK, size=8.5):
    ax.annotate(text, anchor, xytext=offset, textcoords="offset points", fontsize=size, color=color, ha="center",
                va="center", zorder=8)


def zoom(ax, bounds, points, pad):
    """Square inset that frames `points`, with a 0.1 arcsec scale bar."""
    inset = ax.inset_axes(bounds)
    points = np.asarray(points)
    centre = .5 * (points.min(0) + points.max(0))
    span = .5 * (points.max(0) - points.min(0)).max() + pad
    inset.set(xlim=(centre[0] - span, centre[0] + span), ylim=(centre[1] - span, centre[1] + span), aspect="equal",
              xticks=[], yticks=[])
    inset.minorticks_off()
    for spine in inset.spines.values():
        spine.set(lw=.6, color=MUTED)
    ax.indicate_inset_zoom(inset, edgecolor=MUTED, lw=.6, alpha=1)
    x, y = centre[0] - .9 * span, centre[1] - .88 * span
    inset.plot([x, x + .1], [y, y], color=INK, lw=1.2, solid_capstyle="butt")
    inset.text(x + .05, y + .06 * span, '0.1"', ha="center", va="bottom", fontsize=7, color=INK)
    return inset


def draw_image_plane(ax, toy):
    field, measured, true, lens = toy["field"], toy["measured"], toy["true"], toy["lens"]
    limits = dict(xlim=(-30, 50), ylim=(-33, 47))
    cmap = LinearSegmentedColormap.from_list("field", ["#FFFFFF", "#D5D0DC"])
    keep_x = (field.grid >= limits["xlim"][0] - 1) & (field.grid <= limits["xlim"][1] + 1)
    keep_y = (field.grid >= limits["ylim"][0] - 1) & (field.grid <= limits["ylim"][1] + 1)
    ax.imshow(field.kappa[np.ix_(keep_y, keep_x)], origin="lower", cmap=cmap, interpolation="bicubic", zorder=0,
              extent=[field.grid[keep_x][0], field.grid[keep_x][-1], field.grid[keep_y][0], field.grid[keep_y][-1]])
    ax.plot(*toy["critical"].T, color=CRITICAL_COLOR, lw=1.3, zorder=2)
    ax.plot(*measured.T, "o", color=INK, ms=4.2, zorder=4)
    for i, point in enumerate(measured):
        # Number each image outside the critical curve; the two of a close pair are pushed apart.
        out = point / np.hypot(*point)
        others = np.delete(measured, i, 0) - point
        nearest = others[np.argmin(np.hypot(*others.T))]
        if np.hypot(*nearest) < 6:
            out = .6 * out - nearest / np.hypot(*nearest)
            out /= np.hypot(*out)
        ax.text(*(point + 4.2 * out), f"{i + 1}", ha="center", va="center", fontsize=9, color=INK, zorder=5)
    ax.set(**limits, aspect="equal", xlabel=r"$\theta_x$ (arcsec)", ylabel=r"$\theta_y$ (arcsec)")
    ax.text(.035, .965, "(a) Image plane", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold", color=INK)
    ax.text(.965, .04, "shading: unmodeled mass", transform=ax.transAxes, ha="right", fontsize=7.5, color=MUTED)
    right = toy["critical"][np.argmax(toy["critical"][:, 0])]
    ax.annotate("critical curve", right + np.array([-.25, -3]), right + np.array([13, -5.5]), color=CRITICAL_COLOR,
                fontsize=8, ha="center", va="center", arrowprops=dict(arrowstyle="-", color=CRITICAL_COLOR, lw=.6, shrinkA=1, shrinkB=1))
    # Correlation lengths, to compare with the image separations.
    x0, y0 = -26.5, -23
    for k, (length, name, color) in enumerate(((LAMBDA_B, r"$\lambda_{\rm b}$", GRF), (ELL_GP, r"$\ell_{\rm GP}$", GP))):
        y = y0 - 5.2 * k
        ax.plot([x0, x0 + length], [y, y], color=color, lw=2, solid_capstyle="butt", zorder=4)
        ax.text(x0 + length + 1.5, y, name, va="center", color=color, fontsize=9)

    # Zoom on one image: where the lens puts it, its true position and its measured position.
    i = toy["highlight"]
    chain = np.array([lens[i], true[i], measured[i]])
    inset = zoom(ax, [.555, .555, .415, .415], chain, .12)
    inset.add_patch(plt.Circle(true[i], SIGMA_MEAS, fill=False, ls=(0, (3, 2)), lw=.8, color=MEAS))
    arrow(inset, lens[i], true[i], GP)
    arrow(inset, true[i], measured[i], MEAS)
    inset.plot(*lens[i], "+", color=MUTED, ms=7, mew=1.3, zorder=5)
    inset.plot(*true[i], "o", mfc="white", mec=INK, mew=1.1, ms=5.5, zorder=6)
    inset.plot(*measured[i], "o", color=INK, ms=5.5, zorder=6)
    label(inset, r"$\boldsymbol{\theta}_i^{\rm true}$", true[i], away(chain, true[i], 18))
    label(inset, r"$\boldsymbol{\theta}_i^{\rm meas}$", measured[i], away(chain, measured[i], 15))
    label(inset, "lensed\nposition", lens[i], away(chain, lens[i], 17), MUTED, 7.5)
    label(inset, "GP", .5 * (lens[i] + true[i]), beside(lens[i], true[i], chain, 10), GP, 8)
    # The sigma label goes on the rim of the circle, in the widest gap between the two arrows and the label of the point.
    taken = np.sort([np.arctan2(*v[::-1]) for v in (lens[i] - true[i], measured[i] - true[i], away(chain, true[i], 1))])
    gaps = np.diff(np.append(taken, taken[0] + 2 * np.pi))
    angle = taken[np.argmax(gaps)] + .5 * gaps.max()
    rim = np.array([np.cos(angle), np.sin(angle)])
    label(inset, r"$\sigma_{\rm meas}$", true[i] + SIGMA_MEAS * rim, 14 * rim, MEAS, 8.5)
    inset.text(.04, .95, f"Image {i + 1}", transform=inset.transAxes, va="top", fontsize=8, color=INK)


def draw_source_plane(ax, toy):
    source, measured, true, lens = toy["source"], toy["measured"], toy["true"], toy["lens"]
    cusp = np.abs(toy["caustic"]).max()
    ax.plot(*toy["caustic"].T, color=CRITICAL_COLOR, lw=1.3, zorder=2)
    ax.plot(*source, marker="*", color=INK, ms=9, mew=0, zorder=5)
    ax.set(xlim=(-1.2 * cusp, 1.9 * cusp), ylim=(-1.9 * cusp, 1.2 * cusp), aspect="equal",
           xlabel=r"$\beta_x$ (arcsec)", ylabel=r"$\beta_y$ (arcsec)")
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(which="both", left=True)
    ax.text(.035, .965, "(b) Source plane", transform=ax.transAxes, va="top", fontsize=10, fontweight="bold", color=INK)
    right = toy["caustic"][np.argmax(toy["caustic"][:, 0])]
    ax.annotate("caustic", right + np.array([-1.2, .5]), right + np.array([3.2, 2.6]), color=CRITICAL_COLOR, fontsize=8,
                ha="center", arrowprops=dict(arrowstyle="-", color=CRITICAL_COLOR, lw=.6, shrinkA=1, shrinkB=1))

    mapped_true, mapped_measured, mapped_lens = back_project(true), back_project(measured), back_project(lens)
    toy.update(mapped_true=mapped_true, mapped_measured=mapped_measured, mapped_lens=mapped_lens)
    i = toy["highlight"]
    chain = np.array([source, mapped_lens[i], mapped_true[i], mapped_measured[i]])
    inset = zoom(ax, [.50, .03, .47, .47], chain, .09)
    arrow(inset, source, mapped_measured[i], INK, width=.9, head=6, ls=(0, (3, 2)))
    arrow(inset, source, mapped_lens[i], GRF)
    arrow(inset, mapped_lens[i], mapped_true[i], GP)
    arrow(inset, mapped_true[i], mapped_measured[i], MEAS)
    inset.plot(*source, marker="*", color=INK, ms=10, mew=0, zorder=6)
    inset.plot(*mapped_true[i], "o", mfc="white", mec=INK, mew=1.1, ms=5.5, zorder=6)
    inset.plot(*mapped_measured[i], "o", color=INK, ms=5.5, zorder=6)
    label(inset, r"$\boldsymbol{\beta}_k$", source, away(chain, source, 12))
    label(inset, r"$\boldsymbol{\beta}_i$", mapped_measured[i], away(chain, mapped_measured[i], 12))
    label(inset, r"$\boldsymbol{F}_i$", .5 * (source + mapped_measured[i]),
          -beside(source, mapped_measured[i], chain[1:3], 9))
    label(inset, "GRF", .5 * (source + mapped_lens[i]), beside(source, mapped_lens[i], chain, 11), GRF, 8)
    label(inset, "GP", .5 * (mapped_lens[i] + mapped_true[i]), beside(mapped_lens[i], mapped_true[i], chain, 10), GP, 8)
    label(inset, "meas.", .5 * (mapped_true[i] + mapped_measured[i]),
          beside(mapped_true[i], mapped_measured[i], chain, 14), MEAS, 8)
    inset.text(.04, .95, f"Image {i + 1}", transform=inset.transAxes, va="top", fontsize=8, color=INK)


def draw_ellipses(ax, toy, i, half):
    residual = toy["mapped_measured"][i] - toy["source"]
    blocks = [toy[key][i, :, i, :] for key in ("c_meas", "c_gp", "c_grf")]
    for covariance, color in zip(blocks[::-1], (GRF, GP, MEAS)):
        ax.add_patch(ellipse(covariance, color=color, lw=1.5))
    ax.add_patch(ellipse(sum(blocks), color=INK, lw=1.1, ls=(0, (4, 2))))
    ax.plot(0, 0, marker="*", color=INK, ms=8, mew=0, zorder=5)
    ax.plot(*residual, "o", color=INK, ms=4.5, zorder=6)
    ax.set(xlim=(-half, half), ylim=(-half, half), aspect="equal", xlabel=r"$F_x$ (arcsec)")


def draw_matrix(ax, toy):
    n = len(toy["measured"])
    total = (toy["c_meas"] + toy["c_gp"] + toy["c_grf"]).reshape(2 * n, 2 * n)
    scale = np.sqrt(np.diag(total))
    correlation = total / np.outer(scale, scale)
    cmap = LinearSegmentedColormap.from_list("corr", ["#2166AC", "#FFFFFF", "#B2182B"])
    image = ax.imshow(correlation, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    for edge in range(2, 2 * n, 2):
        ax.axhline(edge - .5, color=INK, lw=.5)
        ax.axvline(edge - .5, color=INK, lw=.5)
    ticks = np.arange(n) * 2 + .5
    names = [f"{i + 1}" for i in range(n)]
    ax.set(xticks=ticks, yticks=ticks, xticklabels=names, yticklabels=names, xlabel="Image $j$", ylabel="Image $i$")
    ax.minorticks_off()
    ax.tick_params(length=0, top=False, right=False, pad=4)
    return image, correlation


@plt.rc_context(fname=STYLE, rc={**FIGURE_STYLE, "mathtext.default": "it"})
def main():
    toy = build()
    n = len(toy["measured"])
    print(f"{n} images; chain drawn for image {toy['highlight'] + 1}, near-critical image {toy['near'] + 1}")
    for i in range(n):
        a = toy["jac"][i]
        print(f"  image {i + 1}: measured {toy['measured'][i].round(2)}, det A {np.linalg.det(a):+.3f}, "
              f"eigenvalues {np.linalg.eigvalsh(.5 * (a + a.T)).round(3)}")

    width, height = 8.2, 6.6
    fig = plt.figure(figsize=(width, height))
    box = lambda x, y, w, h: fig.add_axes([x / width, y / height, w / width, h / height])
    top, bottom, big, small = 3.5, .5, 2.95, 2.1
    image_ax = box(.6, top, big, big)
    source_ax = box(width - .6 - big, top, big, big)
    far_ax = box(.6, bottom, small, small)
    near_ax = box(.6 + small + .18, bottom, small, small)
    matrix_ax = box(width - 1.0 - small, bottom, small, small)
    color_ax = box(width - 1.0 + .1, bottom, .09, small)

    draw_image_plane(image_ax, toy)
    draw_source_plane(source_ax, toy)

    # Between the planes: one deflection per image.
    gap_left, gap_right, y_mid = (.6 + big + .12) / width, (width - .6 - big - .12) / width, (top + .5 * big) / height
    fig.add_artist(FancyArrowPatch((gap_left, y_mid), (gap_right, y_mid), transform=fig.transFigure,
                                   arrowstyle="simple,head_length=.9,head_width=1.1,tail_width=.42", mutation_scale=14,
                                   fc="#C9CDD2", ec="none"))
    fig.text(.5, y_mid + .050, "back-projection", ha="center", fontsize=8.5, color=INK)
    fig.text(.5, y_mid + .024, r"$\boldsymbol{\beta}_i=\boldsymbol{\theta}_i-\boldsymbol{\alpha}(\boldsymbol{\theta}_i)$",
             ha="center", fontsize=9.5, color=INK)
    fig.text(.5, y_mid - .040, "areas scale by", ha="center", fontsize=8, color=MUTED)
    fig.text(.5, y_mid - .066, r"$|\det\mathbf{A}_i|$", ha="center", fontsize=9, color=MUTED)

    half = .26
    draw_ellipses(far_ax, toy, toy["highlight"], half)
    draw_ellipses(near_ax, toy, toy["near"], half)
    far_ax.set_ylabel(r"$F_y$ (arcsec)")
    near_ax.tick_params(labelleft=False)
    for ax, i, where in ((far_ax, toy["highlight"], "(c) Image {}\nfar from"), (near_ax, toy["near"], "(d) Image {}\nnear")):
        ax.text(.05, .955, where.format(i + 1) + " the critical curve", transform=ax.transAxes, va="top", fontsize=8,
                color=INK, linespacing=1.3)
    image, correlation = draw_matrix(matrix_ax, toy)
    bar = fig.colorbar(image, cax=color_ax)
    bar.set_label("Correlation of the residuals")
    bar.ax.minorticks_off()
    bar.outline.set_linewidth(.6)

    handles = [Line2D([], [], color=MEAS, lw=1.5), Line2D([], [], color=GP, lw=1.5), Line2D([], [], color=GRF, lw=1.5),
               Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 2)))]
    labels = [r"measurement error, $\mathbf{C}_{\rm meas}$", r"image-plane displacement, $\mathbf{C}_{\rm GP}$",
              r"unmodeled mass, $\mathbf{C}_{\rm GRF}$", r"total, $\mathbf{C}$"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(.6 / width, (bottom + small + .07) / height),
               ncol=2, frameon=False, handlelength=1.8, columnspacing=1.6, borderaxespad=0, fontsize=8)
    matrix_ax.text(0, 1.02, "(e)", transform=matrix_ax.transAxes, fontsize=8.5, color=INK, va="bottom")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(OUT)
    return toy, correlation


if __name__ == "__main__":
    main()
