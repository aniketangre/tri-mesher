"""
app.py — Interactive triangular mesh visualizer.

Usage
-----
    python app.py

Controls
--------
    Resolution slider  — cells along x (1–60)
    Width / Height     — domain dimensions
    Show vertices      — overlay vertex markers
    Color by quality   — shade triangles by aspect ratio (green=good, red=bad)
    Export STL         — save current mesh as a binary STL file
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.patches as mpatches
import matplotlib.widgets as mw

from mesh_generator import generate_equilateral_mesh


# ── Colour palette (Catppuccin Mocha) ────────────────────────────────────────
C = dict(
    bg       = '#1e1e2e',
    surface  = '#181825',
    base     = '#11111b',
    overlay  = '#313244',
    edge     = '#45475a',
    subtext  = '#a6adc8',
    text     = '#cdd6f4',
    accent   = '#89b4fa',   # blue   — sliders, titles
    fill     = '#1d3460',   # mesh triangle fill
    vertex   = '#f38ba8',   # red    — vertex dots
    border   = '#cba6f7',   # mauve  — domain boundary
    green    = '#a6e3a1',   # green  — export success
    yellow   = '#f9e2af',   # yellow — warnings
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _aspect_ratios(mesh) -> np.ndarray:
    v, t = mesh.vertices, mesh.triangles
    el = np.stack([
        np.linalg.norm(v[t[:, 1]] - v[t[:, 0]], axis=1),
        np.linalg.norm(v[t[:, 2]] - v[t[:, 1]], axis=1),
        np.linalg.norm(v[t[:, 0]] - v[t[:, 2]], axis=1),
    ], axis=1)
    return el.max(axis=1) / np.clip(el.min(axis=1), 1e-12, None)


def _cmap():
    """Colormap for quality view: 1.0 (perfect) → green, high AR → red."""
    try:
        return matplotlib.colormaps['RdYlGn_r']
    except AttributeError:
        return plt.cm.get_cmap('RdYlGn_r')


def _style_ax(ax, facecolor=None) -> None:
    ax.set_facecolor(facecolor or C['base'])
    ax.tick_params(colors=C['subtext'], labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C['edge'])


# ── Application ───────────────────────────────────────────────────────────────

class MeshApp:

    def __init__(self) -> None:
        self.nx               = 10
        self.width            = 1.0
        self.height           = 1.0
        self.show_vertices    = False
        self.color_by_quality = False
        self._mesh            = None

        self._build_ui()
        self._refresh()
        plt.show()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(14, 9), facecolor=C['bg'])
        try:
            self.fig.canvas.manager.set_window_title('Triangular Mesh Generator')
        except Exception:
            pass

        # Figure-level title
        self.fig.text(
            0.5, 0.975,
            'Triangular Mesh Generator',
            ha='center', va='top',
            fontsize=15, fontweight='bold',
            color=C['accent'],
        )

        # ── Mesh panel ────────────────────────────────────────────────────────
        self.ax = self.fig.add_axes([0.04, 0.26, 0.53, 0.69])
        _style_ax(self.ax)
        self.ax.set_aspect('equal')

        # Colorbar panel (visible only in quality-colour mode)
        self.ax_cb = self.fig.add_axes([0.585, 0.26, 0.013, 0.69])
        _style_ax(self.ax_cb, facecolor=C['surface'])
        self.ax_cb.set_visible(False)

        # ── Statistics panel ──────────────────────────────────────────────────
        self.ax_st = self.fig.add_axes([0.635, 0.415, 0.335, 0.545])
        self.ax_st.set_facecolor(C['surface'])
        self.ax_st.axis('off')
        # Thin top-border accent line
        self.ax_st.plot([0, 1], [0.97, 0.97], color=C['accent'], linewidth=1.5,
                        transform=self.ax_st.transAxes)
        self.ax_st.text(0.05, 0.99, 'Mesh statistics',
                        transform=self.ax_st.transAxes,
                        va='top', ha='left',
                        fontsize=9, fontweight='bold', color=C['accent'])
        self._stats_txt = self.ax_st.text(
            0.05, 0.90, '',
            transform=self.ax_st.transAxes,
            va='top', ha='left',
            fontsize=9, color=C['text'],
            fontfamily='monospace', linespacing=1.75,
        )

        # ── Export panel ──────────────────────────────────────────────────────
        self.ax_exp = self.fig.add_axes([0.635, 0.310, 0.335, 0.090])
        self.ax_exp.set_facecolor(C['surface'])
        self.ax_exp.axis('off')
        self.ax_exp.plot([0, 1], [0.96, 0.96], color=C['border'], linewidth=1.0,
                         transform=self.ax_exp.transAxes)
        self.ax_exp.text(0.05, 0.99, 'Export',
                         transform=self.ax_exp.transAxes,
                         va='top', fontsize=9, fontweight='bold',
                         color=C['border'])
        self._status_txt = self.ax_exp.text(
            0.05, 0.52, '',
            transform=self.ax_exp.transAxes,
            va='center', fontsize=8,
            color=C['green'], fontfamily='monospace',
        )

        ax_btn = self.fig.add_axes([0.770, 0.318, 0.18, 0.052])
        self._btn_export = mw.Button(
            ax_btn, 'Export STL  ↓',
            color=C['overlay'], hovercolor='#45475a',
        )
        self._btn_export.label.set_color(C['border'])
        self._btn_export.label.set_fontsize(9)
        self._btn_export.on_clicked(self._export_stl)

        # ── Sliders ───────────────────────────────────────────────────────────
        self.sl_nx = self._make_slider(
            [0.08, 0.17, 0.46, 0.028], 'Resolution', 1, 60, self.nx,    step=1)
        self.sl_w  = self._make_slider(
            [0.08, 0.11, 0.46, 0.028], 'Width      ', 0.5, 4.0, self.width)
        self.sl_h  = self._make_slider(
            [0.08, 0.05, 0.46, 0.028], 'Height     ', 0.5, 4.0, self.height)

        # ── Checkboxes ────────────────────────────────────────────────────────
        ax_chk = self.fig.add_axes([0.635, 0.05, 0.335, 0.23],
                                   facecolor=C['surface'])
        ax_chk.plot([0, 1], [0.965, 0.965], color=C['accent'], linewidth=1.0,
                    transform=ax_chk.transAxes)
        ax_chk.text(0.05, 0.985, 'Display options',
                    transform=ax_chk.transAxes,
                    va='top', fontsize=9, fontweight='bold', color=C['accent'])
        ax_chk.axis('off')

        ax_chk_inner = self.fig.add_axes([0.640, 0.065, 0.320, 0.18],
                                         facecolor=C['surface'])
        self.checks = mw.CheckButtons(
            ax_chk_inner,
            ['Show vertices', 'Color by quality'],
            [False, False],
        )
        for lbl in self.checks.labels:
            lbl.set_color(C['text'])
            lbl.set_fontsize(9)

        # Callbacks
        self.sl_nx.on_changed(self._on_slider)
        self.sl_w .on_changed(self._on_slider)
        self.sl_h .on_changed(self._on_slider)
        self.checks.on_clicked(self._on_check)

    def _make_slider(self, rect, label, lo, hi, val, step=None) -> mw.Slider:
        ax = self.fig.add_axes(rect, facecolor=C['overlay'])
        kw: dict = dict(color=C['accent'])
        if step is not None:
            kw['valstep'] = step
        sl = mw.Slider(ax, label, lo, hi, valinit=val, **kw)
        sl.label.set_color(C['subtext']); sl.label.set_fontsize(9)
        sl.valtext.set_color(C['text']); sl.valtext.set_fontsize(9)
        return sl

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _on_slider(self, _) -> None:
        self.nx     = int(self.sl_nx.val)
        self.width  = float(self.sl_w.val)
        self.height = float(self.sl_h.val)
        self._refresh()

    def _on_check(self, label: str) -> None:
        if label == 'Show vertices':
            self.show_vertices    = not self.show_vertices
        elif label == 'Color by quality':
            self.color_by_quality = not self.color_by_quality
        self._refresh()

    def _export_stl(self, _=None) -> None:
        try:
            from PyQt5.QtWidgets import QFileDialog, QApplication
            qapp = QApplication.instance()
            path, _ = QFileDialog.getSaveFileName(
                None, 'Export mesh as STL',
                'mesh.stl', 'STL files (*.stl);;All files (*)',
            )
        except ImportError:
            path = 'mesh.stl'

        if path:
            self._mesh.export_stl(path)
            name = path.split('/')[-1].split('\\')[-1]
            self._status_txt.set_text(f'Saved → {name}')
            self._status_txt.set_color(C['green'])
            self.fig.canvas.draw_idle()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        self._mesh = generate_equilateral_mesh(self.width, self.height, self.nx)
        m = self._mesh

        self.ax.cla()
        _style_ax(self.ax)
        self.ax.set_aspect('equal')
        self.ax.grid(True, color=C['edge'], linewidth=0.3, alpha=0.4, zorder=0)

        verts = m.vertices[m.triangles]   # (M, 3, 2)

        if self.color_by_quality:
            ar   = _aspect_ratios(m)
            cmap = _cmap()
            norm = matplotlib.colors.Normalize(vmin=1.0, vmax=max(ar.max(), 1.01))
            coll = mc.PolyCollection(
                verts,
                facecolors=cmap(norm(ar)),
                edgecolors=C['edge'],
                linewidths=0.35, zorder=1,
            )
            self._draw_colorbar(cmap, ar.min(), ar.max())
        else:
            coll = mc.PolyCollection(
                verts,
                facecolors=C['fill'],
                edgecolors=C['accent'],
                linewidths=0.55, zorder=1,
            )
            self.ax_cb.set_visible(False)

        self.ax.add_collection(coll)

        # Domain boundary — prominent mauve rectangle
        rect = mpatches.FancyBboxPatch(
            (0, 0), self.width, self.height,
            boxstyle='square,pad=0',
            linewidth=2.0, edgecolor=C['border'],
            facecolor='none', zorder=3,
        )
        self.ax.add_patch(rect)

        if self.show_vertices:
            self.ax.scatter(
                m.vertices[:, 0], m.vertices[:, 1],
                s=9, c=C['vertex'], zorder=5, linewidths=0,
            )

        px, py = 0.06 * self.width, 0.06 * self.height
        self.ax.set_xlim(-px, self.width  + px)
        self.ax.set_ylim(-py, self.height + py)
        self.ax.set_xlabel('x', color=C['subtext'], fontsize=9, labelpad=4)
        self.ax.set_ylabel('y', color=C['subtext'], fontsize=9, labelpad=4)

        s = self._mesh.compute_stats()
        self.ax.set_title(
            f'{s.num_triangles} triangles · {s.num_vertices} vertices'
            f'   |   nx = {self.nx}',
            color=C['subtext'], fontsize=9, pad=6,
        )

        self._update_stats(s)
        self.fig.canvas.draw_idle()

    def _draw_colorbar(self, cmap, vmin: float, vmax: float) -> None:
        ax = self.ax_cb
        ax.set_visible(True)
        ax.cla()
        gradient = np.linspace(0, 1, 256).reshape(256, 1)
        ax.imshow(
            gradient[::-1], aspect='auto', cmap=cmap,
            extent=[0, 1, vmin, vmax],
        )
        ax.set_xticks([])
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
        ax.tick_params(colors=C['subtext'], labelsize=7)
        ax.set_ylabel('Aspect ratio', color=C['subtext'], fontsize=8, labelpad=6)
        for sp in ax.spines.values():
            sp.set_color(C['edge'])

    def _update_stats(self, s) -> None:
        self._stats_txt.set_text(
            f"Vertices       {s.num_vertices:>8}\n"
            f"Triangles      {s.num_triangles:>8}\n"
            f"─────────────────────────\n"
            f"Min angle      {s.min_angle:>7.2f}°\n"
            f"Max angle      {s.max_angle:>7.2f}°\n"
            f"Mean angle     {s.mean_angle:>7.2f}°\n"
            f"─────────────────────────\n"
            f"Min area       {s.min_area:>9.5f}\n"
            f"Max area       {s.max_area:>9.5f}\n"
            f"─────────────────────────\n"
            f"Mean AR        {s.mean_aspect_ratio:>7.4f}\n"
            f"Max AR         {s.max_aspect_ratio:>7.4f}\n"
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    MeshApp()
