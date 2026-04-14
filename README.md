# tri-mesher

A Python tool for generating structured equilateral triangular meshes over rectangular domains, with an interactive visualizer and binary STL export.

---

## Overview

Most triangular mesh generators (e.g. Delaunay) produce elements of varying size and shape. **tri-mesher** uses an **offset-row (honeycomb) pattern** to produce a mesh where every interior triangle has angles close to 60° and a near-uniform aspect ratio — regardless of domain size or resolution.

The domain boundary is a perfect rectangle on all four sides. Boundary half-triangles are automatically added at the left and right edges to flush the mesh against the domain walls.

---

## Algorithm

Vertices are placed in alternating rows:

```
Even row  i = 0, 2, 4 …   →   x = j · dx          (nx + 1 vertices)
Odd  row  i = 1, 3, 5 …   →   x = (j − ½) · dx    (nx interior + 2 boundary vertices)
```

The vertical spacing is set to `dy = dx · √3 / 2` (equilateral ideal), then rounded to the nearest integer number of rows and slightly adjusted to fit the domain height exactly.

### Triangle types

| Type | Location | Shape |
|---|---|---|
| Upward / Downward interior | Throughout the mesh | ~60° equilateral |
| Left / Right half-triangles | Left and right edges of each row band | Right triangle |

All triangles are wound **counter-clockwise** (positive signed area).

### Triangle count

```
total = ny × (2 · nx + 1)
```

---

## Project structure

```
mesh-tool/
├── tri-mesh-generator/
│   ├── mesh_generator.py   # Mesh generation + STL export
│   ├── app.py              # Interactive matplotlib visualizer
│   └── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

# Install dependencies
pip install -r tri-mesh-generator/requirements.txt
```

---

## Run

```bash
python tri-mesh-generator/app.py
```

---

## Visualizer controls

| Control | Description |
|---|---|
| **Resolution** slider | Number of cells along x (1 – 60) |
| **Width** slider | Domain width (0.5 – 4.0) |
| **Height** slider | Domain height (0.5 – 4.0) |
| **Show vertices** | Overlay a dot at every mesh node |
| **Color by quality** | Shade each triangle by aspect ratio — green = perfect equilateral, red = distorted |
| **Export STL ↓** | Save the current mesh as a binary STL file |

The **Mesh statistics** panel updates live:

- Vertex and triangle count
- Min / max / mean angles  
- Min / max element area  
- Mean and max aspect ratio (1.0 = perfect equilateral)

---

## Programmatic usage

```python
from tri-mesh-generator.mesh_generator import generate_equilateral_mesh

mesh = generate_equilateral_mesh(width=2.0, height=1.0, nx=20)

print(mesh.vertices.shape)    # (N, 2)
print(mesh.triangles.shape)   # (M, 3)

stats = mesh.compute_stats()
print(f"Triangles : {stats.num_triangles}")
print(f"Min angle : {stats.min_angle:.2f}°")
print(f"Mean AR   : {stats.mean_aspect_ratio:.4f}")

mesh.export_stl("my_mesh.stl")
```

---

## STL export

Meshes are exported as **flat binary STL** files (z = 0, normal = [0, 0, 1]).  
The format is compatible with FEA pre-processors, CAD tools, and mesh viewers such as Meshmixer, Gmsh, and ParaView.

---

## Dependencies

| Package | Version |
|---|---|
| numpy | ≥ 1.21 |
| matplotlib | ≥ 3.4 |
| PyQt5 | ≥ 5.15 |
