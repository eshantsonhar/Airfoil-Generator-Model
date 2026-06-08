from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from airfoil_discovery.cfd.physics import (
    compute_boundary_layer_thickness,
    compute_first_cell_height,
    compute_inflation_layers,
)
from airfoil_discovery.config import MeshConfig


_SURFACE_SIZE_COARSE = 0.0010
_SURFACE_SIZE_LE = 0.00025
_SURFACE_SIZE_TE = 0.00020
_FARFIELD_SIZE = 5.0
_LE_REFINE_RADIUS = 0.05
_WAKE_BOX_XMIN = 0.0
_WAKE_BOX_XMAX_FRACTION = 0.25
_WAKE_BOX_YABS = 0.5
_WAKE_SIZE = 0.0020
_DOMAIN_INLET = 50.0
_DOMAIN_OUTLET = 60.0
_DOMAIN_HALF_HEIGHT = 50.0


@dataclass(slots=True)
class MeshParameters:
    first_cell_height: float
    growth_rate: float
    n_layers: int
    total_bl_thickness: float
    inlet_distance: float
    outlet_distance: float
    half_height: float
    coarse_factor: float
    layer_warning: str | None = None


def compute_mesh_parameters(
    reynolds: float,
    mesh_cfg: MeshConfig,
    coarse_factor: float = 1.0,
) -> MeshParameters:
    growth = min(mesh_cfg.boundary_layer_growth, 1.15)
    first_cell = min(
        compute_first_cell_height(reynolds, y_plus_target=mesh_cfg.y_plus_target),
        mesh_cfg.boundary_layer_first_height,
    ) * coarse_factor
    total_thickness = 1.5 * compute_boundary_layer_thickness(reynolds, x_fraction=0.8) * coarse_factor
    nominal_layers = compute_inflation_layers(
        first_cell / max(coarse_factor, 1e-9),
        growth,
        total_thickness / max(coarse_factor, 1e-9),
        min_layers=mesh_cfg.min_boundary_layer_layers,
        max_layers=mesh_cfg.max_boundary_layer_layers,
    )
    scaled_layers = int(round(nominal_layers / max(coarse_factor, 1e-9)))
    warning = None
    if scaled_layers < mesh_cfg.min_boundary_layer_layers or scaled_layers > mesh_cfg.max_boundary_layer_layers:
        clamped_for_warning = max(mesh_cfg.min_boundary_layer_layers, min(mesh_cfg.max_boundary_layer_layers, scaled_layers))
        warning = (
            f"Inflation layer count clamped for Re={reynolds:.1f}: "
            f"requested {scaled_layers}, using {clamped_for_warning}"
        )
    n_layers = max(mesh_cfg.coarse_layer_min, min(mesh_cfg.coarse_layer_max, scaled_layers))
    bl_thickness = first_cell * (growth**n_layers - 1.0) / max(growth - 1.0, 1e-12)
    if reynolds < 100000.0 or coarse_factor >= 50.0:
        inlet_distance = 10.0
        outlet_distance = 20.0
        half_height = 10.0
    else:
        inlet_distance = max(mesh_cfg.farfield_radius, _DOMAIN_INLET)
        outlet_distance = max(mesh_cfg.wake_length, _DOMAIN_OUTLET)
        half_height = max(mesh_cfg.farfield_radius, _DOMAIN_HALF_HEIGHT)
    return MeshParameters(
        first_cell_height=first_cell,
        growth_rate=growth,
        n_layers=n_layers,
        total_bl_thickness=bl_thickness,
        inlet_distance=inlet_distance,
        outlet_distance=outlet_distance,
        half_height=half_height,
        coarse_factor=coarse_factor,
        layer_warning=warning,
    )


def build_geo_script(
    coords: np.ndarray,
    reynolds: float,
    mesh_cfg: MeshConfig,
    coarse_factor: float = 1.0,
) -> str:
    unique: list[np.ndarray] = [coords[0]]
    for point in coords[1:]:
        if np.linalg.norm(point - unique[-1]) > 1e-9:
            unique.append(point)
    if np.linalg.norm(unique[-1] - unique[0]) < 1e-9:
        unique.pop()

    params = compute_mesh_parameters(reynolds, mesh_cfg, coarse_factor=coarse_factor)
    n_pts = len(unique)
    lines: list[str] = [
        'SetFactory("Built-in");',
        "// domain_rationale: inlet=50c upstream, outlet=60c downstream, top_bottom=50c for low-Re external-flow isolation",
        "// boundary_layer_rationale: y_plus_target<=0.5, 45-65 nominal layers, growth<=1.15 for wall-resolved transition prediction",
        "// boundary_layer_note: explicit BoundaryLayer field is omitted in the generated mesh because the local Gmsh build stalls on this geometry; near-wall sizing is enforced through surface refinement instead",
        f"// coarse_factor: {params.coarse_factor:.3f}",
        f"// first_cell_height: {params.first_cell_height:.6e}",
        f"// boundary_layer_growth: {params.growth_rate:.4f}",
        f"// boundary_layer_layers: {params.n_layers}",
        f"// boundary_layer_thickness: {params.total_bl_thickness:.6e}",
    ]
    if params.layer_warning:
        lines.append(f"// warning: {params.layer_warning}")
    lines.extend(["", "// airfoil points"])
    surface_size_coarse = 0.008 if reynolds < 100000.0 else _SURFACE_SIZE_COARSE
    surface_size_le = 0.002 if reynolds < 100000.0 else _SURFACE_SIZE_LE
    surface_size_te = 0.003 if reynolds < 100000.0 else _SURFACE_SIZE_TE

    for idx, (x, y) in enumerate(unique, start=1):
        if x <= _LE_REFINE_RADIUS:
            size = surface_size_le * coarse_factor
        elif x >= 0.94:
            size = surface_size_te * coarse_factor
        else:
            size = surface_size_coarse * coarse_factor
        lines.append(f"Point({idx}) = {{{x:.10f}, {y:.10f}, 0.0, {size:.6f}}};")

    airfoil_line_ids: list[int] = []
    lines.append("")
    for idx in range(1, n_pts):
        line_id = idx
        airfoil_line_ids.append(line_id)
        lines.append(f"Line({line_id}) = {{{idx}, {idx + 1}}};")
    closing_line_id = n_pts
    airfoil_line_ids.append(closing_line_id)
    lines.append(f"Line({closing_line_id}) = {{{n_pts}, 1}};")
    airfoil_ids = ",".join(str(i) for i in airfoil_line_ids)
    lines.extend(
        [
            f"Curve Loop(2) = {{{airfoil_ids}}};",
            "",
        ]
    )

    p_ib = n_pts + 1
    p_ob = n_pts + 2
    p_ot = n_pts + 3
    p_it = n_pts + 4
    farfield_size = 2.0 if coarse_factor >= 50.0 else _FARFIELD_SIZE * coarse_factor
    lines.extend(
        [
            f"Point({p_ib}) = {{{-params.inlet_distance:.2f}, {-params.half_height:.2f}, 0.0, {farfield_size:.2f}}};",
            f"Point({p_ob}) = {{{params.outlet_distance:.2f}, {-params.half_height:.2f}, 0.0, {farfield_size:.2f}}};",
            f"Point({p_ot}) = {{{params.outlet_distance:.2f}, {params.half_height:.2f}, 0.0, {farfield_size:.2f}}};",
            f"Point({p_it}) = {{{-params.inlet_distance:.2f}, {params.half_height:.2f}, 0.0, {farfield_size:.2f}}};",
            "Line(200001) = {%d, %d};" % (p_ib, p_ob),
            "Line(200002) = {%d, %d};" % (p_ob, p_ot),
            "Line(200003) = {%d, %d};" % (p_ot, p_it),
            "Line(200004) = {%d, %d};" % (p_it, p_ib),
            "Curve Loop(3) = {200001, 200002, 200003, 200004};",
            "Plane Surface(1) = {3, 2};",
            "",
        ]
    )
    if coarse_factor >= 50.0 and reynolds >= 100000.0:
        lines.extend(
            [
                f"// L0 fast mesh omits distance/background fields for reliable interactive meshing.",
                f"// Field[3].SizeMin = {surface_size_le * coarse_factor:.4f};",
                f"// Field[4].XMax = {max(15.0, params.outlet_distance * _WAKE_BOX_XMAX_FRACTION):.3f};",
                "",
            ]
        )
    else:
        lines.extend(
            [
            "Field[2] = Distance;",
            f"Field[2].CurvesList = {{{airfoil_ids}}};",
            "Field[2].Sampling = 300;",
            "Field[3] = Threshold;",
            "Field[3].InField = 2;",
            f"Field[3].SizeMin = {surface_size_le * coarse_factor:.4f};",
            f"Field[3].SizeMax = {surface_size_coarse * coarse_factor:.4f};",
            "Field[3].DistMin = 0.0;",
            f"Field[3].DistMax = {_LE_REFINE_RADIUS:.4f};",
            "",
            "Field[4] = Box;",
            f"Field[4].XMin = {_WAKE_BOX_XMIN:.3f};",
            f"Field[4].XMax = {max(15.0, params.outlet_distance * _WAKE_BOX_XMAX_FRACTION):.3f};",
            f"Field[4].YMin = {-_WAKE_BOX_YABS:.3f};",
            f"Field[4].YMax = {_WAKE_BOX_YABS:.3f};",
            "Field[4].ZMin = -1.0;",
            "Field[4].ZMax = 1.0;",
            f"Field[4].VIn = {_WAKE_SIZE * coarse_factor:.4f};",
            f"Field[4].VOut = {_FARFIELD_SIZE * coarse_factor:.2f};",
            "",
            "Field[5] = Min;",
            "Field[5].FieldsList = {3, 4};",
            "Background Field = 5;",
            "",
            ]
        )
    lines.extend(
        [
            'Physical Surface("fluid") = {1};',
            'Physical Curve("farfield") = {200001, 200002, 200003, 200004};',
            f'Physical Curve("airfoil") = {{{airfoil_ids}}};',
            "",
            "Mesh.Algorithm = 5;",
            "Mesh.RecombineAll = 0;",
            "Mesh.Smoothing = 5;",
            "Mesh.CharacteristicLengthExtendFromBoundary = 0;",
            "Mesh.CharacteristicLengthFromCurvature = 1;",
            f"Mesh.MinimumCurveNodes = {max(20, int(math.ceil(20 / max(coarse_factor, 1e-9))))};",
        ]
    )
    return "\n".join(lines)


@dataclass
class FidelityParams:
    level: str
    coarse_factor: float
    y_plus_target: float


class MeshFidelityManager:
    """
    Standardized mesh fidelity registry for multi-fidelity ASO.
    Ensures consistent grid escalation across optimization campaigns.
    """
    REGISTRY = {
        "L0": FidelityParams("L0", coarse_factor=100.0, y_plus_target=5.0),
        "L1": FidelityParams("L1", coarse_factor=50.0, y_plus_target=2.0),
        "L2": FidelityParams("L2", coarse_factor=20.0, y_plus_target=1.0),
    }

    @classmethod
    def get_params(cls, level: str) -> FidelityParams:
        return cls.REGISTRY.get(level, cls.REGISTRY["L1"])


def generate_aso_mesh(
    coords: np.ndarray,
    reynolds: float,
    mesh_cfg: MeshConfig,
    level: str,
) -> str:
    """
    High-level entry point for generating ASO-specific mesh fidelity levels.
    """
    params = MeshFidelityManager.get_params(level)

    # Temporarily override mesh_cfg y_plus for the fidelity level
    original_y_plus = mesh_cfg.y_plus_target
    mesh_cfg.y_plus_target = params.y_plus_target

    geo_script = build_geo_script(
        coords=coords,
        reynolds=reynolds,
        mesh_cfg=mesh_cfg,
        coarse_factor=params.coarse_factor,
    )

    mesh_cfg.y_plus_target = original_y_plus  # Restore
    return geo_script
