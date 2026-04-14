"""
mesh_generator.py — Structured equilateral triangular mesh over a rectangle.

Vertex layout
-------------
  • Even rows  i = 0, 2, 4, …  →  (nx + 1) vertices at  x = j · dx
  • Odd  rows  i = 1, 3, 5, …  →  (nx + 2) vertices:
        j = 0       : x = 0              (left-edge boundary vertex)
        j = 1..nx   : x = (j − ½) · dx  (interior offset vertices)
        j = nx + 1  : x = width          (right-edge boundary vertex)

Adding boundary vertices on odd rows gives every row a vertex at x = 0 and
x = width, so all four sides of the domain are flush rectangles.  The two
extra half-triangles per row band are right triangles; interior triangles
remain near-equilateral (60°).

All triangles wound counter-clockwise (positive signed area).
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ── data types ────────────────────────────────────────────────────────────────

@dataclass
class MeshStats:
    num_vertices:      int
    num_triangles:     int
    min_angle:         float
    max_angle:         float
    mean_angle:        float
    min_area:          float
    max_area:          float
    mean_aspect_ratio: float
    max_aspect_ratio:  float


class TriangularMesh:
    """Vertex coordinates + triangle connectivity."""

    def __init__(self, vertices: np.ndarray, triangles: np.ndarray) -> None:
        self.vertices  = vertices    # (N, 2)  float64
        self.triangles = triangles   # (M, 3)  int64,  CCW

    # ── quality metrics ───────────────────────────────────────────────────────

    def compute_stats(self) -> MeshStats:
        v, t = self.vertices, self.triangles
        v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        e01, e02, e12 = v1 - v0, v2 - v0, v2 - v1

        l01 = np.linalg.norm(e01, axis=1)
        l02 = np.linalg.norm(e02, axis=1)
        l12 = np.linalg.norm(e12, axis=1)

        cos0 = np.clip(np.einsum('ij,ij->i', e01,  e02)  / (l01 * l02), -1, 1)
        cos1 = np.clip(np.einsum('ij,ij->i', -e01, e12)  / (l01 * l12), -1, 1)
        cos2 = np.clip(np.einsum('ij,ij->i', e02,  e12)  / (l02 * l12), -1, 1)
        angles = np.degrees(np.arccos(np.stack([cos0, cos1, cos2], axis=1)))

        areas = 0.5 * (e01[:, 0] * e02[:, 1] - e01[:, 1] * e02[:, 0])

        el = np.stack([l01, l02, l12], axis=1)
        ar = el.max(axis=1) / np.clip(el.min(axis=1), 1e-12, None)

        return MeshStats(
            num_vertices      = len(v),
            num_triangles     = len(t),
            min_angle         = float(angles.min()),
            max_angle         = float(angles.max()),
            mean_angle        = float(angles.mean()),
            min_area          = float(areas.min()),
            max_area          = float(areas.max()),
            mean_aspect_ratio = float(ar.mean()),
            max_aspect_ratio  = float(ar.max()),
        )

    # ── export ────────────────────────────────────────────────────────────────

    def export_stl(self, filepath: str) -> None:
        """
        Write the mesh as a flat (z = 0) binary STL file.

        The STL records are built with numpy structured arrays for speed.
        All triangles are oriented with normal (0, 0, +1).

        Parameters
        ----------
        filepath : destination path, e.g. 'mesh.stl'
        """
        n = len(self.triangles)

        # 3-D coordinates (z = 0)
        v3 = np.column_stack([self.vertices,
                               np.zeros(len(self.vertices))]).astype(np.float32)

        v0 = v3[self.triangles[:, 0]]
        v1 = v3[self.triangles[:, 1]]
        v2 = v3[self.triangles[:, 2]]

        # Binary STL record layout (50 bytes each):
        #   normal  3 × float32
        #   v0      3 × float32
        #   v1      3 × float32
        #   v2      3 × float32
        #   attr    uint16
        dtype = np.dtype([
            ('normal', np.float32, 3),
            ('v0',     np.float32, 3),
            ('v1',     np.float32, 3),
            ('v2',     np.float32, 3),
            ('attr',   np.uint16),
        ])
        records          = np.zeros(n, dtype=dtype)
        records['normal'] = [0.0, 0.0, 1.0]
        records['v0']     = v0
        records['v1']     = v1
        records['v2']     = v2

        with open(filepath, 'wb') as f:
            header = b'STL exported by tri-mesh-generator'
            f.write(header.ljust(80, b'\x00'))
            f.write(np.uint32(n).tobytes())
            f.write(records.tobytes())


# ── mesh generation ───────────────────────────────────────────────────────────

def generate_equilateral_mesh(
    width:  float,
    height: float,
    nx:     int,
) -> TriangularMesh:
    """
    Generate a structured triangular mesh over [0, width] × [0, height].

    Interior triangles have angles close to 60°.  Boundary half-triangles
    fill the left and right strips so the domain has a perfect rectangular
    outline on all four sides.

    Parameters
    ----------
    width, height : domain dimensions  (positive floats)
    nx            : number of cells along x  (≥ 1)
    """
    nx = max(1, int(nx))
    dx = width / nx
    ny = max(1, round(height / (dx * np.sqrt(3.0) / 2.0)))
    dy = height / ny

    # ── vertices ──────────────────────────────────────────────────────────────
    row_start: list[int] = []
    coords: list[list[float]] = []

    for i in range(ny + 1):
        row_start.append(len(coords))
        y = i * dy
        if i % 2 == 0:
            # nx + 1 vertices at x = j·dx
            for j in range(nx + 1):
                coords.append([j * dx, y])
        else:
            # nx + 2 vertices: left boundary, interior offsets, right boundary
            coords.append([0.0, y])
            for j in range(nx):
                coords.append([(j + 0.5) * dx, y])
            coords.append([width, y])

    vertices = np.array(coords, dtype=np.float64)

    def vid(row: int, col: int) -> int:
        return row_start[row] + col

    # ── triangles (all CCW) ───────────────────────────────────────────────────
    #
    #  Even→Odd band  (bottom row = even, top row = odd):
    #
    #    even[0]─even[1]─even[2]─ … ─even[nx]
    #      |  \  ↑  /  \  ↑  /            |
    #    odd[0] odd[1] odd[2] … odd[nx] odd[nx+1]
    #
    #  Left / right strips use half-triangles to maintain the flush boundary.
    #
    #  Odd→Even band is the mirror: odd row on bottom, even row on top.
    #
    tris: list[list[int]] = []

    for i in range(ny):
        if i % 2 == 0:
            # ── Even → Odd ────────────────────────────────────────────────
            # Left boundary half-triangle
            tris.append([vid(i, 0), vid(i+1, 1), vid(i+1, 0)])
            # Interior upward triangles
            for j in range(nx):
                tris.append([vid(i, j), vid(i, j+1), vid(i+1, j+1)])
            # Interior downward triangles
            for j in range(1, nx):
                tris.append([vid(i, j), vid(i+1, j+1), vid(i+1, j)])
            # Right boundary half-triangle
            tris.append([vid(i, nx), vid(i+1, nx+1), vid(i+1, nx)])
        else:
            # ── Odd → Even ────────────────────────────────────────────────
            # Left boundary half-triangle
            tris.append([vid(i, 0), vid(i, 1), vid(i+1, 0)])
            # Interior downward triangles
            for j in range(1, nx + 1):
                tris.append([vid(i, j), vid(i+1, j), vid(i+1, j-1)])
            # Interior upward triangles
            for j in range(1, nx):
                tris.append([vid(i, j), vid(i, j+1), vid(i+1, j)])
            # Right boundary half-triangle
            tris.append([vid(i, nx), vid(i, nx+1), vid(i+1, nx)])

    return TriangularMesh(vertices, np.array(tris, dtype=np.int64))
