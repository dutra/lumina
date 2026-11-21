"""Vector schematic of paper/main.tex, Eqs. measurement_distribution--likelihood_single.

Illustrative affine mapping only; contours have unit Mahalanobis radius, not
68% joint coverage. No cluster data or fitted posterior is represented.
Run with the project's lenstronomy Python. Outputs PDF only.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, ConnectionPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/figures/likelihood_two_factors.pdf'
BLUE, RED, INK, GRAY = '#2274A5', '#AD4672', '#222C36', '#65717B'


def ellipse(ax, center, cov, color, ls='-', fill=True):
    v, u = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(u[1, 1], u[0, 1]))
    ax.add_patch(Ellipse(center, 2*np.sqrt(v[1]), 2*np.sqrt(v[0]), angle=angle,
                         facecolor=color if fill else 'none', alpha=.09 if fill else 1,
                         edgecolor='none' if fill else color, lw=1.6, linestyle=ls))
    if fill:
        ellipse(ax, center, cov, color, ls, False)


def note(ax, text, point, offset, color=INK, ha='left'):
    ax.annotate(text, point, xytext=offset, textcoords='offset points',
                ha=ha, va='center', fontsize=11, color=color)


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'mathtext.fontset': 'dejavusans', 'axes.grid': False,
                         'pdf.fonttype': 42})
    fig = plt.figure(figsize=(12, 8.6), facecolor='white')
    left = fig.add_axes([.055, .43, .37, .40])
    right = fig.add_axes([.575, .43, .37, .40])
    for ax, symbol, plane in [(left, r'\theta', 'Image plane'), (right, r'\beta', 'Source plane')]:
        ax.set(xlim=(-2, 2), ylim=(-1.55, 1.55), aspect='equal', xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set_visible(False)
        origin = (-1.83, -1.30)
        ax.annotate('', (-1.12, -1.30), origin, arrowprops={'arrowstyle': '->', 'color': GRAY})
        ax.annotate('', (-1.83, -.65), origin, arrowprops={'arrowstyle': '->', 'color': GRAY})
        ax.text(-1.07, -1.30, '$'+symbol+'_1$', va='center', color=GRAY)
        ax.text(-1.83, -.56, '$'+symbol+'_2$', ha='center', color=GRAY)
        ax.text(.5, -.015, plane, transform=ax.transAxes, ha='center', fontsize=13)

    true = np.array([-.35, .12])
    meas = np.array([.30, .57])
    beta_k = np.array([-.40, -.12])
    beta_i = np.array([.38, .34])
    A = np.array([[.65, .22], [.22, .38]])
    S = .65**2*np.eye(2)
    Cm = np.array([[.65, -.15], [-.15, .29]])
    Cmeas = A@S@A.T
    beta_meas = beta_i + A@(meas-true)
    ellipse(left, true, S, BLUE)
    left.plot(*true, 'o', mfc='white', mec=INK, ms=8, mew=1.7, zorder=5)
    left.plot(*meas, 'x', color=BLUE, ms=10, mew=2, zorder=5)
    left.annotate('', meas, true, arrowprops={'arrowstyle': '->', 'color': BLUE, 'shrinkA': 7, 'shrinkB': 7})
    note(left, r'$\mathbf{\theta}_i^{\rm true}$'+'\nlatent; integrated out', true, (-7, -76), ha='center')
    note(left, r'$\mathbf{\theta}_i^{\rm meas}$'+'\nobserved', meas, (22, -4), BLUE)
    left.text(-1.65, 1.21, r'$\mathbf{S}_i=\sigma_{{\rm meas},i}^{2}\mathbf{I}$', color=BLUE, fontsize=13)
    left.text(-1.65, .94, 'Measurement scatter about the true position', color=BLUE, fontsize=9)

    ellipse(right, beta_k, Cm, RED)
    ellipse(right, beta_i, Cmeas, BLUE, '--', False)
    right.plot(*beta_k, '*', color=RED, ms=15, zorder=6)
    right.plot(*beta_i, 'o', mfc='white', mec=INK, ms=8, mew=1.7, zorder=6)
    right.plot(*beta_meas, 'x', color=BLUE, ms=9, mew=2, zorder=6)
    right.annotate('', beta_i, beta_k, arrowprops={'arrowstyle': '->', 'color': RED, 'shrinkA': 9, 'shrinkB': 7})
    note(right, r'$\mathbf{\beta}_k$'+'\nsampled family source', beta_k, (-4, -67), RED, 'center')
    note(right, r'$\mathbf{\beta}_i$'+'\nbackprojected true image', beta_i, (24, -30))
    note(right, r'$\mathbf{\beta}(\mathbf{\theta}_i^{\rm meas})$', beta_meas, (0, 20), BLUE, 'center')
    right.text(-1.65, 1.21, r'$\mathbf{C}_{{\rm model},ii}$', color=RED, fontsize=13)
    right.text(-1.65, .94, 'Model scatter about the family source', color=RED, fontsize=9)
    right.text(-.90, .38, r'$\mathbf{F}_i$', color=RED, fontsize=13)

    fig.add_artist(ConnectionPatch(xyA=(.435, .64), coordsA=fig.transFigure,
                    xyB=(.565, .64), coordsB=fig.transFigure, arrowstyle='->',
                    lw=1.3, color=GRAY))
    fig.text(.5, .66, 'Lens mapping', ha='center', color=GRAY, fontsize=10)
    fig.text(.5, .91, r'$\mathbf{\beta}_i=\mathbf{\theta}_i^{\rm true}-\mathbf{\alpha}(\mathbf{\theta}_i^{\rm true})$',
             ha='center', fontsize=15)
    fig.text(.5, .866, 'Map each candidate true image through the lens model', ha='center', color=GRAY)

    fig.text(.25, .375, r'$p(\mathbf{\theta}_i^{\rm meas}\mid\mathbf{\theta}_i^{\rm true},\mathbf{\Theta})$',
             ha='center', color=BLUE, fontsize=14)
    fig.text(.25, .331, r'$=\mathcal{N}(\mathbf{\theta}_i^{\rm meas}-\mathbf{\theta}_i^{\rm true};\,\mathbf{S}_i)$',
             ha='center', color=BLUE, fontsize=13)
    fig.text(.75, .375, r'$p(\mathbf{\theta}_i^{\rm true}\mid\mathbf{\Theta})$', ha='center', color=RED, fontsize=14)
    fig.text(.75, .331, r'$=\mathcal{N}(\mathbf{\beta}_i-\mathbf{\beta}_k;\,\mathbf{C}_{{\rm model},ii})\,|\det\mathbf{A}_i|$',
             ha='center', color=RED, fontsize=13)
    fig.text(.75, .294, r'$|\det\mathbf{A}_i|$: source density $\rightarrow$ local image density', ha='center', color=GRAY, fontsize=10)
    fig.text(.5, .239, r'$\mathcal{L}_i=\int d^2\theta_i^{\rm true}\;[\mathrm{measurement\ factor}]\;[\mathrm{model\ factor}]$', ha='center', fontsize=15)
    fig.text(.5, .201, 'At every latent position, multiply the two factors; then integrate over that position.', ha='center', fontsize=10)
    fig.add_artist(plt.Line2D([.06, .94], [.175, .175], transform=fig.transFigure, color='#CCD2D6', lw=.8))
    fig.text(.5, .142, r'Locally: $\mathbf{C}_{{\rm meas},ii}=\mathbf{A}_i\mathbf{S}_i\mathbf{A}_i^{\rm T}$ (blue dashed ellipse),'
             r' $\quad\mathbf{A}_i=\partial\mathbf{\beta}/\partial\mathbf{\theta}$', ha='center', fontsize=11)
    fig.text(.5, .102, r'After linearization: $\mathbf{C}=\mathbf{C}_{\rm meas}+\mathbf{C}_{\rm model}$,'
             r' $\quad\mathbf{C}_{\rm model}=\mathbf{C}_{\rm GRF}+\mathbf{C}_{\rm GP}$', ha='center', fontsize=11)
    fig.text(.5, .063, 'For multiple images, model covariance couples residuals through off-diagonal blocks; integrate jointly.', ha='center', color=GRAY, fontsize=10)
    fig.text(.5, .027, 'Schematic local geometry. Contours show unit Mahalanobis radius; the two planes use independent scales.', ha='center', color=GRAY, fontsize=9)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)
    print(OUT)


if __name__ == '__main__':
    main()
