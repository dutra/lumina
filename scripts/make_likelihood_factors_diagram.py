"""Diagram of the two factors of the image-position likelihood of one image, and of the covariance of all images.

    L_i = int d^2 theta_true  N(theta_meas - theta_true; sigma_meas^2 I)  x  N(F_i(theta_true); C_model,i) |det A_i|

Panel (a) is the image plane, where the measured and the true position live: each factor is drawn as a function of the
true position (one contour at one standard deviation), with the region where their product is large, which the
integral adds up.  Panel (b) is the source plane, where the second factor is evaluated: the true position mapped back,
the source position of the family, the residual between them and the covariance of that residual.  The lower row is
the covariance of the whole family, term by term.

The toy lens and its field are those of make_likelihood_diagram.py; the measurement error is larger and the draw of the
errors is another one.  Writes paper/figures/position_likelihood_factors.pdf.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from matplotlib.patheffects import withStroke
from scipy.optimize import root

import make_likelihood_diagram as toy_lens
from make_likelihood_diagram import GP, GRF, INK, MEAS, MUTED, STYLE, back_project, parametric
from kappaccino.plotting.style import APJ_TEXT_WIDTH_IN, BLUE, FIGURE_STYLE, save_page

OUT = Path(__file__).resolve().parents[1] / "paper" / "figures" / "position_likelihood_factors.pdf"
IMAGE = 3                               # the image whose likelihood is drawn (index in the toy family)
SIGMA_MEAS = 0.07                       # arcsec; larger than in the other diagram so both factors shape the product
SEED_ERRORS = 2612                      # a draw where the two factors overlap and the true position lies between them
MODEL = BLUE                            # the model factor; the measurement factor keeps the colour of C_meas
SHADE, SHADE_INK = "#C3C7CC", "#5E646B"  # the integrand, and text about the integral
# One colour per variable, shared by its marker and its label: the true position keeps its colour in both planes.
TRUE, SOURCE = "#7B3FA0", "#B07A00"
ONE_SIGMA = np.exp(-.5)                 # density contour at one standard deviation, relative to the peak
HALO = [withStroke(linewidth=2.6, foreground="white")]


def covariance_ellipse(centre, covariance, **style):
    values, vectors = np.linalg.eigh(covariance)
    angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
    return Ellipse(centre, 2 * np.sqrt(max(values[1], 0)), 2 * np.sqrt(max(values[0], 0)), angle=angle, fill=False, **style)


def note(ax, text, xy, offset, color=INK, size=8.5, leader=True, halo=True, **kwargs):
    """Text at an offset (points) from a position, with a thin leader line."""
    line = dict(arrowstyle="-", color=color, lw=.5, shrinkA=2, shrinkB=kwargs.pop("shrink", 4)) if leader else None
    ax.annotate(text, xy, xytext=offset, textcoords="offset points", fontsize=size, color=color, ha=kwargs.pop("ha", "center"),
                va=kwargs.pop("va", "center"), zorder=9, path_effects=HALO if halo else None, linespacing=1.3, arrowprops=line, **kwargs)


def frame(ax, centre, half, title):
    ax.set(xlim=(centre[0] - half, centre[0] + half), ylim=(centre[1] - half, centre[1] + half), xticks=[], yticks=[], aspect="equal")
    ax.minorticks_off()
    for spine in ax.spines.values():
        spine.set(lw=.6, color=INK)
    ax.text(.04, .96, title, transform=ax.transAxes, va="top", fontsize=10, fontweight="bold", color=INK)
    x, y = centre[0] - .9 * half, centre[1] - .88 * half
    ax.plot([x, x + .1], [y, y], color=INK, lw=1.2, solid_capstyle="butt", zorder=8)
    ax.text(x + .05, y + .04 * half, '0.1"', ha="center", va="bottom", fontsize=7.5, color=INK)


def factors_on_grid(toy, predicted, half=.6, n=601):
    """Both factors and their product as functions of the true position, each relative to its peak."""
    measured, source = toy["measured"][IMAGE], toy["source"]
    model_cov = toy["c_gp"][IMAGE, :, IMAGE, :] + toy["c_grf"][IMAGE, :, IMAGE, :]
    centre = .5 * (measured + predicted)
    axis = np.linspace(-half, half, n)
    xx, yy = np.meshgrid(centre[0] + axis, centre[1] + axis)
    points = np.stack([xx, yy], -1)
    deflection, hessian = parametric(points)
    residual = points - deflection - source
    jac = np.eye(2) - hessian
    det = jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0]
    model = np.exp(-.5 * np.einsum("...i,ij,...j", residual, np.linalg.inv(model_cov), residual)) * np.abs(det)
    measurement = np.exp(-.5 * ((points - measured)**2).sum(-1) / SIGMA_MEAS**2)
    product = model * measurement
    return xx, yy, measurement, model / model.max(), product / product.max()


def draw_image_plane(ax, toy, predicted, grid):
    """Each factor as a function of the true position, and the region where their product is large."""
    measured, true = toy["measured"][IMAGE], toy["true"][IMAGE]
    xx, yy, measurement, model, product = grid
    centre = .5 * (measured + predicted) + np.array([.0, -.01])
    frame(ax, centre, .37, "(a) Image plane")
    region = ax.contourf(xx, yy, product, levels=[ONE_SIGMA, 1.01], colors=[SHADE], zorder=1)
    ax.contour(xx, yy, model, levels=[ONE_SIGMA], colors=[MODEL], linewidths=1.6, zorder=3)
    ax.contour(xx, yy, measurement, levels=[ONE_SIGMA], colors=[MEAS], linewidths=1.6, zorder=4)
    ax.plot(*predicted, marker="*", color=SOURCE, ms=13, mew=0, zorder=6)   # the image of beta_k: same star, same colour
    ax.plot(*measured, "o", color=INK, ms=5.5, zorder=7)
    ax.plot(*true, "o", mfc="white", mec=TRUE, mew=1.5, ms=5.8, zorder=7)
    note(ax, r"$\boldsymbol{\theta}_i^{\rm meas}$", measured, (-41, 1), INK, 10, leader=False)
    note(ax, r"$\boldsymbol{\theta}_i^{\rm true}$", true, (15, -12), TRUE, 10, leader=False, halo=False)
    low = yy[product > ONE_SIGMA].min()
    blob = np.array([xx[product > ONE_SIGMA].mean(), low + .012])
    note(ax, "integrand", blob, (-60, 0), SHADE_INK, 8.5, leader=False)
    # Each contour carries the word of its factor under the equation.
    note(ax, "measurement", measured + np.array([0, SIGMA_MEAS]), (0, 9), MEAS, 8.5, leader=False)
    inside = model > ONE_SIGMA
    row = np.abs(yy - predicted[1] + .03) < .002
    note(ax, "lens model", np.array([xx[inside & row].min(), predicted[1] - .03]), (-28, 0), MODEL, 8.5, leader=False)
    note(ax, "model image", predicted, (0, -14), SOURCE, 8, leader=False)
    return region


def draw_source_plane(ax, toy, region):
    """The second factor where it is evaluated: the residual of the mapped true position and its covariance."""
    source, true = toy["source"], toy["true"][IMAGE]
    mapped_true = back_project(true)
    c_gp, c_grf = (toy[key][IMAGE, :, IMAGE, :] for key in ("c_gp", "c_grf"))
    frame(ax, source + np.array([-.02, -.055]), .30, "(b) Source plane")
    for path in region.get_paths():                      # the true positions that count, mapped back
        for polygon in path.to_polygons():
            ax.add_patch(Polygon(back_project(polygon), closed=True, fc=SHADE, ec="none", zorder=1))
    ax.add_patch(covariance_ellipse(source, c_grf, color=GRF, lw=1.1, zorder=2))
    ax.add_patch(covariance_ellipse(source, c_gp, color=GP, lw=1.1, zorder=2))
    ax.add_patch(covariance_ellipse(source, c_gp + c_grf, color=MODEL, lw=1.8, zorder=3))
    # The residual: the tip stops at the rim of the open circle and is drawn over everything else.
    ax.add_patch(FancyArrowPatch(source, mapped_true, arrowstyle="-|>", mutation_scale=8, color=INK, lw=1.1, shrinkA=0, shrinkB=4.4,
                                 joinstyle="miter", capstyle="butt", zorder=8))
    ax.plot(*source, marker="*", color=SOURCE, ms=13, mew=0, zorder=9)    # over the tail of the arrow
    ax.plot(*mapped_true, "o", mfc="white", mec=TRUE, mew=1.5, ms=5.8, zorder=7)
    label = dict(leader=False, size=10, halo=False)
    note(ax, r"$\boldsymbol{\beta}_k$", source, (-12, -1), SOURCE, 9, leader=False, halo=False)
    note(ax, r"$\boldsymbol{\beta}_i$", mapped_true, (-15, -3), TRUE, **label)


def equation(fig, y, pieces, size=11, pad=.016):
    """Lay out coloured pieces of one equation on a centred line; returns the horizontal extent of each piece."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [fig.text(0, y, text, color=color, fontsize=size, va="center", ha="left") for text, color in pieces]
    widths = [t.get_window_extent(renderer).width / fig.bbox.width for t in texts]
    x, extents = .5 - .5 * (sum(widths) + pad * (len(widths) - 1)), []
    for text, width in zip(texts, widths):
        text.set_x(x)
        extents.append((x, x + width))
        x += width + pad
    return extents


def bracket(fig, extent, y, color, text, centre=None):
    """Bracket under one piece of the equation with what it means; returns where an arrow can leave the text."""
    x0, x1 = extent
    fig.add_artist(Line2D([x0, x0, x1, x1], [y + .007, y, y, y + .007], color=color, lw=.9, transform=fig.transFigure))
    centre = .5 * (x0 + x1) if centre is None else centre
    fig.text(centre, y - .011, text, color=color, fontsize=8.5, ha="center", va="top")


def draw_matrices(fig, boxes, color_box, toy):
    """The covariance of every image of the family, term by term, on one colour scale."""
    n = len(toy["measured"])
    terms = [toy[key].reshape(2 * n, 2 * n) for key in ("c_meas", "c_gp", "c_grf")]
    terms.append(sum(terms))
    terms = [np.sign(term) * np.sqrt(np.abs(term)) for term in terms]          # in arcsec, so the small terms stay visible
    names = [(r"$\mathbf{C}_{\rm meas}$", MEAS), (r"$\mathbf{C}_{\rm GP}$", GP), (r"$\mathbf{C}_{\rm GRF}$", GRF), (r"$\mathbf{C}$", INK)]
    limit = np.abs(terms[-1]).max()
    cmap = LinearSegmentedColormap.from_list("covariance", ["#2166AC", "#FFFFFF", "#B2182B"])
    for k, (box, term, (name, color)) in enumerate(zip(boxes, terms, names)):
        ax = fig.add_axes(box)
        image = ax.imshow(term, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        for edge in range(2, 2 * n, 2):
            ax.axhline(edge - .5, color=INK, lw=.4)
            ax.axvline(edge - .5, color=INK, lw=.4)
        ticks = np.arange(n) * 2 + .5
        ax.set(xticks=ticks, yticks=ticks, xticklabels=[f"{i + 1}" for i in range(n)],
               yticklabels=[f"{i + 1}" for i in range(n)] if k == 0 else [])
        ax.minorticks_off()
        ax.tick_params(length=0, top=False, right=False, pad=2, labelsize=7.5)
        for spine in ax.spines.values():
            spine.set(lw=.8, color=color)
        ax.text(.5, 1.05, name, transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color=color)
        if k == 0:
            ax.set_ylabel("image", fontsize=8, labelpad=2)
    for left, right, sign in zip(boxes[:-1], boxes[1:], ("+", "+", "=")):
        fig.text(.5 * (left[0] + left[2] + right[0]), left[1] + .5 * left[3], sign, ha="center", va="center", fontsize=13, color=INK)
    bar_ax = fig.add_axes(color_box)
    bar = fig.colorbar(image, cax=bar_ax, ticks=[-limit, 0, limit])
    bar.ax.set_yticklabels([f"$-${limit:.2f}", "0", f"{limit:.2f}"], fontsize=7.5)
    bar.ax.minorticks_off()
    bar.outline.set_linewidth(.6)
    bar_ax.text(.5, 1.05, "arcsec", transform=bar_ax.transAxes, ha="center", va="bottom", fontsize=7.5, color=INK)


@plt.rc_context(fname=STYLE, rc={**FIGURE_STYLE, "mathtext.default": "it"})
def main():
    toy_lens.SIGMA_MEAS = SIGMA_MEAS
    toy = toy_lens.build(SEED_ERRORS)
    predicted = root(lambda theta: back_project(theta) - toy["source"], toy["measured"][IMAGE], tol=1e-12).x
    print(f"image {IMAGE + 1}: det A {np.linalg.det(toy['jac'][IMAGE]):+.3f}, measured - model image {(toy['measured'][IMAGE] - predicted).round(3)}")

    width, height = APJ_TEXT_WIDTH_IN, 5.95
    fig = plt.figure(figsize=(width, height))
    box = lambda x, y, w, h: [x / width, y / height, w / width, h / height]
    side, gap, top = 2.72, 1.16, 2.42
    left = .5 * (width - 2 * side - gap)
    image_ax = fig.add_axes(box(left, top, side, side))
    source_ax = fig.add_axes(box(left + side + gap, top, side, side))
    region = draw_image_plane(image_ax, toy, predicted, factors_on_grid(toy, predicted))
    draw_source_plane(source_ax, toy, region)

    # The likelihood of one image: the integral over the true position and one colour per factor.
    y_eq = (height - .28) / height
    extents = equation(fig, y_eq, [
        (r"$\mathcal{L}_i\;=$", INK),
        (r"$\int d^2\theta_i^{\rm true}$", SHADE_INK),
        (r"$\mathcal{N}\!\left(\boldsymbol{\theta}_i^{\rm meas}-\boldsymbol{\theta}_i^{\rm true};\,\sigma_{{\rm meas},i}^2\mathbf{I}\right)$", MEAS),
        (r"$\times$", INK),
        (r"$\mathcal{N}\!\left(\boldsymbol{F}_i(\boldsymbol{\theta}_i^{\rm true});\,\mathbf{C}_{{\rm model},i}\right)\,\left|\det\mathbf{A}_i\right|$", MODEL)])
    y_bracket = y_eq - .034
    bracket(fig, extents[1], y_bracket, SHADE_INK, "marginalized")
    bracket(fig, extents[2], y_bracket, MEAS, "measurement")
    bracket(fig, extents[4], y_bracket, MODEL, "lens model")
    # Between the planes: one deflection per image, and the change of area.
    x0, x1, y_mid = (left + side + .12) / width, (left + side + gap - .12) / width, (top + .5 * side) / height
    fig.add_artist(FancyArrowPatch((x0, y_mid), (x1, y_mid), transform=fig.transFigure, fc="#C9CDD2", ec="none", mutation_scale=14,
                                   arrowstyle="simple,head_length=.9,head_width=1.1,tail_width=.42"))
    fig.text(.5, y_mid + .098, "lens model\nmaps back", ha="center", va="center", fontsize=8.5, color=MUTED, linespacing=1.3)
    fig.text(.5, y_mid + .042, r"$\mathbf{A}_i\mathbf{A}_i^{\rm T}$", ha="center", va="center", fontsize=10, color=INK)
    fig.text(.5, y_mid - .042, r"$|\det\mathbf{A}_i|$", ha="center", va="center", fontsize=10, color=INK)

    # The ellipses of (b), named by colour, along the bottom of the panel beside the scale bar.
    centre = (left + side + gap + .63 * side) / width
    lines = [((top + .2) / height, [(r"$\mathbf{C}_{{\rm model},i}$", MODEL), (r"$=$", INK), (r"$\mathbf{C}_{\rm GRF}$", GRF), (r"$+$", INK),
                                   (r"$\mathbf{C}_{\rm GP}$", GP)]),
             ((top + .46) / height, [(r"$\boldsymbol{F}_i$", INK), (r"$=$", INK), (r"$\boldsymbol{\beta}_i$", TRUE), (r"$-$", INK),
                                    (r"$\boldsymbol{\beta}_k$", SOURCE)])]
    starts = []
    for y, key in lines:
        extents = equation(fig, y, key, size=9.5, pad=.008)
        starts.append((fig.texts[-len(key):], extents))
    # Both lines are set with their equals signs under one another.
    for texts, extents in starts:
        shift = centre - .5 * sum(extents[1])
        for text in texts:
            text.set_x(text.get_position()[0] + shift)

    # The whole family: the covariance of the residuals after the true positions are integrated out.
    y_head = (top - .43) / height
    fig.text(left / width, y_head, "(c) All images", ha="left", va="center", fontsize=10, fontweight="bold", color=INK)
    fig.text(1 - left / width, y_head, r"$\mathcal{L}_{\rm pos}\simeq\mathcal{N}\!\left(\boldsymbol{F}(\boldsymbol{\theta}^{\rm meas});\,\mathbf{C}\right)\,"
             r"\langle\prod_i|\det\mathbf{A}_i|\rangle$", ha="right", va="center", fontsize=10, color=INK)
    size, step, x_first, y_matrix = 1.16, 1.56, .62, .38
    boxes = [box(x_first + k * step, y_matrix, size, size) for k in range(4)]
    draw_matrices(fig, boxes, box(x_first + 3 * step + size + .12, y_matrix, .08, size), toy)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_page(fig, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
