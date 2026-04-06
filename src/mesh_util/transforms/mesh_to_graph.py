from functools import partial
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch
from torch_geometric.data import Data

from src.helpers.custom_types import MeshNodeType
from src.helpers.qol import safe_concatenate
from src.tasks.domains.mesh_wrapper import MeshWrapper
from src.tasks.features.feature_provider import FeatureProvider

EPS = 1e-12

# =========================
# Helper utilities (numpy)
# =========================
def _boundary_nodes(mesh) -> np.ndarray:
    """Sorted unique global indices of boundary nodes."""
    return np.asarray(mesh.boundary_nodes(), dtype=np.int64)


def _boundary_edge_indices(wrapped_mesh: MeshWrapper) -> np.ndarray:
    """Boundary edge indices into wrapped_mesh.mesh_edges columns (1D)."""
    return np.asarray(wrapped_mesh.boundary_edges, dtype=np.int64)


def _edge_pairs_from_boundary_edges(wrapped_mesh: MeshWrapper) -> np.ndarray:
    """Boundary edge endpoint pairs, shape (2, Kb)."""
    be = _boundary_edge_indices(wrapped_mesh)
    me = np.asarray(wrapped_mesh.mesh_edges, dtype=np.int64)  # (2, E)
    if me.ndim != 2 or me.shape[0] != 2:
        raise ValueError(f"Expected mesh_edges shape (2, E), got {me.shape}")
    return me[:, be]


def _boundary_vertex_mask(wrapped_mesh: MeshWrapper) -> np.ndarray:
    """Boolean mask of boundary vertices, shape (V,)."""
    V = wrapped_mesh.num_vertices
    mask = np.zeros(V, dtype=bool)
    b = _boundary_nodes(wrapped_mesh.mesh)
    mask[b] = True
    return mask


def _degree_from_edges(num_vertices: int, mesh_edges_2xE: np.ndarray) -> np.ndarray:
    """Degree for each vertex given undirected edges (2, E)."""
    flat = mesh_edges_2xE.reshape(-1)
    return np.bincount(flat, minlength=num_vertices).astype(np.float32)


def _boundary_degree(num_vertices: int, boundary_edge_pairs_2xK: np.ndarray) -> np.ndarray:
    """Boundary degree for each vertex given boundary edges (2, K)."""
    flat = boundary_edge_pairs_2xK.reshape(-1)
    return np.bincount(flat, minlength=num_vertices).astype(np.float32)


def _bbox_normalize_positions(pos: np.ndarray) -> np.ndarray:
    """Center and scale positions by bbox diagonal."""
    pmin = pos.min(axis=0)
    pmax = pos.max(axis=0)
    center = 0.5 * (pmin + pmax)
    diag = np.linalg.norm(pmax - pmin) + EPS
    return (pos - center) / diag


def _as_dimxNb_boundary_normals(wrapped_mesh: MeshWrapper) -> np.ndarray:
    """Return boundary normals in shape (dim, Nb)."""
    bn = np.asarray(wrapped_mesh.boundary_vertex_normals, dtype=np.float32)
    dim = wrapped_mesh.mesh.dim()
    if bn.ndim != 2:
        raise ValueError(f"boundary_vertex_normals must be 2D, got {bn.shape}")
    if bn.shape[0] == dim:
        return bn
    if bn.shape[1] == dim:
        return bn.T
    raise ValueError(f"Unrecognized boundary normals shape {bn.shape} for dim={dim}")


def _full_boundary_normals_VxD(wrapped_mesh: MeshWrapper) -> np.ndarray:
    """Full vertex normals (V, dim). Interior vertices are zeros."""
    V = wrapped_mesh.num_vertices
    dim = wrapped_mesh.mesh.dim()
    full = np.zeros((V, dim), dtype=np.float32)

    b_nodes = _boundary_nodes(wrapped_mesh.mesh)  # (Nb,)
    bn_dimxNb = _as_dimxNb_boundary_normals(wrapped_mesh)  # (dim, Nb)

    if bn_dimxNb.shape[1] != len(b_nodes):
        raise ValueError("boundary_nodes and boundary_vertex_normals length mismatch")

    full[b_nodes] = bn_dimxNb.T  # (Nb, dim)
    return full


def _normal_variation_features(wrapped_mesh: MeshWrapper) -> Dict[str, np.ndarray]:
    """Neighbor normal angle mean/max for each vertex; non-boundary gets 0."""
    edges = np.asarray(wrapped_mesh.mesh_edges, dtype=np.int64)  # (2, E)
    V = wrapped_mesh.num_vertices
    n = _full_boundary_normals_VxD(wrapped_mesh)  # (V, dim)

    u = edges[0]
    v = edges[1]
    nu = n[u]
    nv = n[v]

    mu = np.linalg.norm(nu, axis=1) > 0
    mv = np.linalg.norm(nv, axis=1) > 0
    m = mu & mv

    dots = np.sum(nu[m] * nv[m], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.arccos(dots).astype(np.float32)  # (E_valid,)

    uu = u[m]
    vv = v[m]

    sum_ang = np.zeros(V, dtype=np.float32)
    cnt = np.zeros(V, dtype=np.float32)
    max_ang = np.zeros(V, dtype=np.float32)

    np.add.at(sum_ang, uu, ang)
    np.add.at(sum_ang, vv, ang)
    np.add.at(cnt, uu, 1.0)
    np.add.at(cnt, vv, 1.0)
    np.maximum.at(max_ang, uu, ang)
    np.maximum.at(max_ang, vv, ang)

    mean_ang = sum_ang / np.maximum(cnt, 1.0)
    return {"normal_var_mean": mean_ang, "normal_var_max": max_ang}


def _boundary_triangles_3xF(wrapped_mesh: MeshWrapper) -> Optional[np.ndarray]:
    """For 3D tet meshes: boundary triangles (3, F) from mesh.facets[:, boundary_facets]."""
    if wrapped_mesh.mesh.dim() != 3:
        return None
    bf = np.asarray(wrapped_mesh.mesh.boundary_facets(), dtype=np.int64)  # (F,)
    tris = np.asarray(wrapped_mesh.mesh.facets[:, bf], dtype=np.int64)  # (3, F)
    return tris


def _mean_gaussian_curvature_on_boundary(wrapped_mesh: MeshWrapper) -> Dict[str, np.ndarray]:
    """Mean curvature magnitude and Gaussian curvature on boundary surface; (V,) each. Non-boundary -> 0."""
    V = wrapped_mesh.num_vertices
    if wrapped_mesh.mesh.dim() != 3:
        z = np.zeros(V, dtype=np.float32)
        return {"mean_curvature": z, "gaussian_curvature": z}

    tris = _boundary_triangles_3xF(wrapped_mesh)
    if tris is None:
        z = np.zeros(V, dtype=np.float32)
        return {"mean_curvature": z, "gaussian_curvature": z}

    pos = np.asarray(wrapped_mesh.vertex_positions, dtype=np.float64)  # (V,3)
    i0, i1, i2 = tris[0], tris[1], tris[2]
    p0, p1, p2 = pos[i0], pos[i1], pos[i2]

    # triangle areas
    e10 = p1 - p0
    e20 = p2 - p0
    cross = np.cross(e10, e20)
    area = 0.5 * (np.linalg.norm(cross, axis=1) + EPS)  # (F,)

    # vertex area (barycentric 1/3)
    A = np.zeros(V, dtype=np.float64)
    np.add.at(A, i0, area / 3.0)
    np.add.at(A, i1, area / 3.0)
    np.add.at(A, i2, area / 3.0)
    A = np.maximum(A, EPS)

    # corner angles for gaussian curvature (angle deficit)
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        u = b - a
        v = c - a
        du = np.linalg.norm(u, axis=1) + EPS
        dv = np.linalg.norm(v, axis=1) + EPS
        dots = np.sum(u * v, axis=1) / (du * dv)
        dots = np.clip(dots, -1.0, 1.0)
        return np.arccos(dots)

    ang0 = _angle(p0, p1, p2)
    ang1 = _angle(p1, p2, p0)
    ang2 = _angle(p2, p0, p1)

    sum_ang = np.zeros(V, dtype=np.float64)
    np.add.at(sum_ang, i0, ang0)
    np.add.at(sum_ang, i1, ang1)
    np.add.at(sum_ang, i2, ang2)

    K = (2.0 * np.pi - sum_ang) / A  # (V,)

    # cotan Laplacian for mean curvature magnitude
    def _cot(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        cr = np.linalg.norm(np.cross(u, v), axis=1) + EPS
        return np.sum(u * v, axis=1) / cr

    cot0 = _cot(p1 - p0, p2 - p0)  # opposite edge (i1,i2)
    cot1 = _cot(p2 - p1, p0 - p1)  # opposite edge (i2,i0)
    cot2 = _cot(p0 - p2, p1 - p2)  # opposite edge (i0,i1)

    L = np.zeros((V, 3), dtype=np.float64)

    def _add_edge_contrib(a_idx: np.ndarray, b_idx: np.ndarray, w_ab: np.ndarray) -> None:
        np.add.at(L, a_idx, w_ab[:, None] * (pos[b_idx] - pos[a_idx]))
        np.add.at(L, b_idx, w_ab[:, None] * (pos[a_idx] - pos[b_idx]))

    _add_edge_contrib(i1, i2, cot0)
    _add_edge_contrib(i2, i0, cot1)
    _add_edge_contrib(i0, i1, cot2)

    H = (0.5 * np.linalg.norm(L, axis=1)) / A  # (V,)

    bmask = _boundary_vertex_mask(wrapped_mesh)
    H_out = np.zeros(V, dtype=np.float32)
    K_out = np.zeros(V, dtype=np.float32)
    H_out[bmask] = H[bmask].astype(np.float32)
    K_out[bmask] = K[bmask].astype(np.float32)

    return {"mean_curvature": H_out, "gaussian_curvature": K_out}


def _dihedral_angle_per_mesh_edge(wrapped_mesh: MeshWrapper) -> np.ndarray:
    """Dihedral angle for each mesh edge (undirected), shape (E,). 2D -> zeros."""
    edges = np.asarray(wrapped_mesh.mesh_edges, dtype=np.int64)  # (2, E)
    E = edges.shape[1]
    if wrapped_mesh.mesh.dim() != 3:
        return np.zeros(E, dtype=np.float32)

    tris = _boundary_triangles_3xF(wrapped_mesh)
    if tris is None:
        return np.zeros(E, dtype=np.float32)

    pos = np.asarray(wrapped_mesh.vertex_positions, dtype=np.float64)  # (V,3)

    # boundary face normals
    i0, i1, i2 = tris[0], tris[1], tris[2]
    p0, p1, p2 = pos[i0], pos[i1], pos[i2]
    fn = np.cross(p1 - p0, p2 - p0)
    fn = fn / (np.linalg.norm(fn, axis=1)[:, None] + EPS)  # (F,3)

    # map undirected edge -> list of boundary face normals
    edge2ns: Dict[Tuple[int, int], List[np.ndarray]] = {}
    F = tris.shape[1]
    for f in range(F):
        a, b, c = int(i0[f]), int(i1[f]), int(i2[f])
        n = fn[f]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edge2ns.setdefault(key, []).append(n)

    dih = np.zeros(E, dtype=np.float32)
    for e in range(E):
        a = int(edges[0, e])
        b = int(edges[1, e])
        key = (a, b) if a < b else (b, a)
        ns = edge2ns.get(key, [])
        if len(ns) >= 2:
            dot = float(np.clip(np.dot(ns[0], ns[1]), -1.0, 1.0))
            dih[e] = math.acos(dot)
    return dih


def _sharp_edge_flag_from_dihedral(dih: np.ndarray, threshold_degrees: float = 30.0) -> np.ndarray:
    th = math.radians(threshold_degrees)
    return (np.abs(dih) > th).astype(np.float32)


# =========================
# Graph builders
# =========================
def mesh_to_graph(
    wrapped_mesh: MeshWrapper,
    node_feature_names: List[str],
    edge_feature_names: List[str],
    feature_provider: Optional[FeatureProvider],
    node_type: MeshNodeType = "element",
    add_self_edges: bool = True,
) -> Data:
    """
    Generates an observation graph from a finite element problem and a sizing field.
    The graph is used as input for the GNN-based supervised learning algorithm.
    Args:
        wrapped_mesh: MeshWrapper containing the mesh data as required for, e.g., the mesh connectivity and
            element midpoints
        feature_provider: Class containing problem-specific information, such as boundary conditions and material for a FEM,
            or inlet position for the molding task.
            May be None if a task has no features.
        node_feature_names: Names of the features to use for the element nodes
        edge_feature_names: Names of the features to use for the edges between the elements in the observation graph
        node_type: The type of the nodes in the graph. Either "element" for a graph over mesh elements, or
                "vertex" for a graph over mesh vertices
    Returns:
    """
    if node_type == "element":
        node_features = get_mesh_element_features(
            wrapped_mesh, feature_provider=feature_provider, node_feature_names=node_feature_names
        )
        edge_attr, edge_index = get_mesh_element_edges(
            wrapped_mesh, edge_feature_names=edge_feature_names, add_self_edges=add_self_edges
        )
    elif node_type == "vertex":
        node_features = get_mesh_vertex_features(
            wrapped_mesh, feature_provider=feature_provider, node_feature_names=node_feature_names
        )
        edge_attr, edge_index = get_mesh_vertex_edges(
            wrapped_mesh, edge_feature_names=edge_feature_names, add_self_edges=add_self_edges
        )
    else:
        raise ValueError(f"Node type {node_type=} not supported")

    graph_dict = {
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    # print(f"node_features.shape:{node_features.shape}")
    # print(f"edge_attr.shape:{edge_attr.shape}")

    graph = Data(**graph_dict)
    return graph


def get_mesh_element_edges(
    wrapped_mesh: MeshWrapper, edge_feature_names: List[str], add_self_edges: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get edges for a mesh. The edge features are computed based on the edge_feature_names.
    Args:
        wrapped_mesh:
        edge_feature_names:
        add_self_edges:

    Returns:

    """
    node_neighbors = torch.tensor(wrapped_mesh.element_neighbors, dtype=torch.long)
    node_positions = torch.tensor(wrapped_mesh.element_midpoints, dtype=torch.float32)

    src_nodes = torch.cat([node_neighbors[0], node_neighbors[1]], dim=0)
    dest_nodes = torch.cat([node_neighbors[1], node_neighbors[0]], dim=0)

    if add_self_edges:
        num_nodes = wrapped_mesh.num_elements
        src_nodes = torch.cat([src_nodes, torch.arange(num_nodes)], dim=0)
        dest_nodes = torch.cat([dest_nodes, torch.arange(num_nodes)], dim=0)

    edge_features = []
    if "distance_vector" in edge_feature_names:
        distance_vectors = node_positions[dest_nodes] - node_positions[src_nodes]
        edge_features.extend(list(distance_vectors.T))

    if "euclidean_distance" in edge_feature_names:
        euclidean_distances = torch.norm(node_positions[dest_nodes] - node_positions[src_nodes], dim=1)
        edge_features.append(euclidean_distances)

    edge_index = torch.vstack((src_nodes, dest_nodes)).long()
    edge_attr = torch.stack(edge_features, dim=1)  # shape: [num_edges, num_features]

    return edge_attr, edge_index


def get_mesh_vertex_edges(
    wrapped_mesh: MeshWrapper, edge_feature_names: List[str], add_self_edges: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get edges for a vertex graph. Edge features based on edge_feature_names.
    """
    node_neighbors = torch.tensor(wrapped_mesh.mesh_edges, dtype=torch.long)  # (2, E)
    node_positions = torch.tensor(wrapped_mesh.vertex_positions, dtype=torch.float32)  # (V, dim)

    src_nodes = torch.cat([node_neighbors[0], node_neighbors[1]], dim=0)  # (2E,)
    dest_nodes = torch.cat([node_neighbors[1], node_neighbors[0]], dim=0)  # (2E,)

    if add_self_edges:
        num_nodes = wrapped_mesh.num_vertices
        src_nodes = torch.cat([src_nodes, torch.arange(num_nodes)], dim=0)
        dest_nodes = torch.cat([dest_nodes, torch.arange(num_nodes)], dim=0)

    edge_features: List[torch.Tensor] = []

    if "distance_vector" in edge_feature_names:
        distance_vectors = node_positions[dest_nodes] - node_positions[src_nodes]
        edge_features.extend(list(distance_vectors.T))

    if "euclidean_distance" in edge_feature_names:
        euclidean_distances = torch.norm(node_positions[dest_nodes] - node_positions[src_nodes], dim=1)
        edge_features.append(euclidean_distances)

    # Original AMBER-style boundary edge curvature (signed angle between boundary vertex normals)
    if "edge_curvature" in edge_feature_names:
        boundary_edge_curvatures = np.asarray(wrapped_mesh.boundary_edge_curvatures, dtype=np.float32)
        edge_curvatures = np.zeros(wrapped_mesh.mesh_edges.shape[1], dtype=np.float32)
        edge_curvatures[_boundary_edge_indices(wrapped_mesh)] = boundary_edge_curvatures

        # repeat for both directions
        edge_curvatures = np.concatenate((edge_curvatures, edge_curvatures), axis=0)

        # add 0s for self edges
        if add_self_edges:
            edge_curvatures = np.concatenate(
                (edge_curvatures, np.zeros(wrapped_mesh.num_vertices, dtype=np.float32)), axis=0
            )

        edge_features.append(torch.tensor(edge_curvatures, dtype=torch.float32))

    # =====================
    # New edge features
    # =====================
    edges_np = np.asarray(wrapped_mesh.mesh_edges, dtype=np.int64)  # (2, E)
    E = edges_np.shape[1]

    # boundary_edge_flag: 1 if edge is boundary edge (undirected), expanded to directed (+self)
    if "boundary_edge_flag" in edge_feature_names:
        undirected = np.zeros(E, dtype=np.float32)
        undirected[_boundary_edge_indices(wrapped_mesh)] = 1.0
        directed = np.concatenate([undirected, undirected], axis=0)
        if add_self_edges:
            directed = np.concatenate([directed, np.zeros(wrapped_mesh.num_vertices, dtype=np.float32)], axis=0)
        edge_features.append(torch.tensor(directed, dtype=torch.float32))

    # edge_length_over_sqrt_hihj: |e| / sqrt(h_i h_j)
    if "edge_length_over_sqrt_hihj" in edge_feature_names:
        from src.mesh_util.sizing_field_util import get_sizing_field

        h = get_sizing_field(mesh=wrapped_mesh, mesh_node_type="vertex").astype(np.float32)
        pos = np.asarray(wrapped_mesh.vertex_positions, dtype=np.float32)
        a = edges_np[0]
        b = edges_np[1]
        elen = np.linalg.norm(pos[a] - pos[b], axis=1).astype(np.float32)
        denom = np.sqrt(np.maximum(h[a] * h[b], EPS)).astype(np.float32)
        undirected = elen / denom

        directed = np.concatenate([undirected, undirected], axis=0)
        if add_self_edges:
            directed = np.concatenate([directed, np.zeros(wrapped_mesh.num_vertices, dtype=np.float32)], axis=0)
        edge_features.append(torch.tensor(directed, dtype=torch.float32))

    # delta_log_sizing_field: log(h_dest) - log(h_src) for directed edges (+self=0)
    if "delta_log_sizing_field" in edge_feature_names:
        from src.mesh_util.sizing_field_util import get_sizing_field

        h = get_sizing_field(mesh=wrapped_mesh, mesh_node_type="vertex").astype(np.float32)
        logh = np.log(np.maximum(h, EPS)).astype(np.float32)
        src_np = src_nodes.cpu().numpy()
        dst_np = dest_nodes.cpu().numpy()
        dlog = (logh[dst_np] - logh[src_np]).astype(np.float32)
        edge_features.append(torch.tensor(dlog, dtype=torch.float32))

    # dihedral_angle / sharp_edge_flag (3D only; else zeros)
    if ("dihedral_angle" in edge_feature_names) or ("sharp_edge_flag" in edge_feature_names):
        dih = _dihedral_angle_per_mesh_edge(wrapped_mesh)  # (E,)
        dih_dir = np.concatenate([dih, dih], axis=0)
        if add_self_edges:
            dih_dir = np.concatenate([dih_dir, np.zeros(wrapped_mesh.num_vertices, dtype=np.float32)], axis=0)

        if "dihedral_angle" in edge_feature_names:
            edge_features.append(torch.tensor(dih_dir, dtype=torch.float32))

        if "sharp_edge_flag" in edge_feature_names:
            sharp_u = _sharp_edge_flag_from_dihedral(dih, threshold_degrees=30.0)  # (E,)
            sharp_dir = np.concatenate([sharp_u, sharp_u], axis=0)
            if add_self_edges:
                sharp_dir = np.concatenate([sharp_dir, np.zeros(wrapped_mesh.num_vertices, dtype=np.float32)], axis=0)
            edge_features.append(torch.tensor(sharp_dir, dtype=torch.float32))

    edge_index = torch.vstack((src_nodes, dest_nodes)).long()
    edge_attr = (
        torch.stack(edge_features, dim=1)
        if edge_features
        else torch.zeros((edge_index.shape[1], 0), dtype=torch.float32)
    )
    return edge_attr, edge_index


def get_mesh_element_features(
    wrapped_mesh: MeshWrapper, feature_provider: Optional[FeatureProvider], node_feature_names: List[str]
) -> torch.Tensor:
    """
    Extracts general element features for a given mesh and optional problem-specific features.
    """
    general_element_features: List[np.ndarray] = []

    if "x_position" in node_feature_names:
        general_element_features.append(wrapped_mesh.element_midpoints[:, 0])
    if "y_position" in node_feature_names:
        general_element_features.append(wrapped_mesh.element_midpoints[:, 1])
    if "z_position" in node_feature_names:
        assert wrapped_mesh.dim() == 3, "z_position is only available for 3D meshes"
        general_element_features.append(wrapped_mesh.element_midpoints[:, 2])

    if "sizing_field" in node_feature_names or "log_sizing_field" in node_feature_names:
        from src.mesh_util.sizing_field_util import get_sizing_field

        h = get_sizing_field(mesh=wrapped_mesh, mesh_node_type="element").astype(np.float32)
        if "sizing_field" in node_feature_names:
            general_element_features.append(h)
        if "log_sizing_field" in node_feature_names:
            general_element_features.append(np.log(np.maximum(h, EPS)).astype(np.float32))

    if (
        "x_position_norm" in node_feature_names
        or "y_position_norm" in node_feature_names
        or "z_position_norm" in node_feature_names
    ):
        mp = np.asarray(wrapped_mesh.element_midpoints, dtype=np.float32)
        mpn = _bbox_normalize_positions(mp)
        if "x_position_norm" in node_feature_names:
            general_element_features.append(mpn[:, 0])
        if "y_position_norm" in node_feature_names:
            general_element_features.append(mpn[:, 1])
        if mpn.shape[1] == 3 and "z_position_norm" in node_feature_names:
            general_element_features.append(mpn[:, 2])

    if "simplex_volume" in node_feature_names:
        general_element_features.append(np.asarray(wrapped_mesh.simplex_volumes, dtype=np.float32))

    general_element_features_arr = np.array(general_element_features).T if general_element_features else None

    if feature_provider is not None:
        element_fem_features = feature_provider.get_element_features(wrapped_mesh=wrapped_mesh)
        element_features = safe_concatenate([general_element_features_arr, element_fem_features], axis=1)
    else:
        element_features = general_element_features_arr

    if element_features is None:
        return torch.zeros((wrapped_mesh.num_elements, 0), dtype=torch.float32)
    return torch.tensor(element_features, dtype=torch.float32)


def get_mesh_vertex_features(
    wrapped_mesh: MeshWrapper, feature_provider: Optional[FeatureProvider], node_feature_names: List[str]
) -> torch.Tensor:
    """
    Extracts general vertex features for a given mesh and optional problem-specific features.
    """
    general_vertex_features: List[np.ndarray] = []

    # raw positions (as in original)
    if "x_position" in node_feature_names:
        general_vertex_features.append(wrapped_mesh.p[0])
    if "y_position" in node_feature_names:
        general_vertex_features.append(wrapped_mesh.p[1])
    if "z_position" in node_feature_names:
        assert wrapped_mesh.dim() == 3, "z_position is only available for 3D meshes"
        general_vertex_features.append(wrapped_mesh.p[2])

    # degree (fixed implementation: bincount)
    if "degree" in node_feature_names:
        deg = _degree_from_edges(wrapped_mesh.num_vertices, np.asarray(wrapped_mesh.mesh_edges, dtype=np.int64))
        general_vertex_features.append(deg)

    # sizing field and log sizing field
    if "sizing_field" in node_feature_names or "log_sizing_field" in node_feature_names:
        from src.mesh_util.sizing_field_util import get_sizing_field

        h = get_sizing_field(mesh=wrapped_mesh, mesh_node_type="vertex").astype(np.float32)
        if "sizing_field" in node_feature_names:
            general_vertex_features.append(h)

        if "log_sizing_field" in node_feature_names:
            general_vertex_features.append(np.log(np.maximum(h, EPS)).astype(np.float32))

    # =====================
    # New vertex features
    # =====================
    if "is_boundary" in node_feature_names:
        general_vertex_features.append(_boundary_vertex_mask(wrapped_mesh).astype(np.float32))

    if "boundary_degree" in node_feature_names:
        b_pairs = _edge_pairs_from_boundary_edges(wrapped_mesh)  # (2, Kb)
        general_vertex_features.append(_boundary_degree(wrapped_mesh.num_vertices, b_pairs))

    if (
        "x_position_norm" in node_feature_names
        or "y_position_norm" in node_feature_names
        or "z_position_norm" in node_feature_names
    ):
        pos = np.asarray(wrapped_mesh.vertex_positions, dtype=np.float32)  # (V, dim)
        pn = _bbox_normalize_positions(pos)
        if "x_position_norm" in node_feature_names:
            general_vertex_features.append(pn[:, 0])
        if "y_position_norm" in node_feature_names:
            general_vertex_features.append(pn[:, 1])
        if pn.shape[1] == 3 and "z_position_norm" in node_feature_names:
            general_vertex_features.append(pn[:, 2])

    if "normal_x" in node_feature_names or "normal_y" in node_feature_names or "normal_z" in node_feature_names:
        nf = _full_boundary_normals_VxD(wrapped_mesh)  # (V, dim)
        if "normal_x" in node_feature_names:
            general_vertex_features.append(nf[:, 0])
        if "normal_y" in node_feature_names:
            general_vertex_features.append(nf[:, 1])
        if nf.shape[1] == 3 and "normal_z" in node_feature_names:
            general_vertex_features.append(nf[:, 2])

    if "normal_var_mean" in node_feature_names or "normal_var_max" in node_feature_names:
        nv = _normal_variation_features(wrapped_mesh)
        if "normal_var_mean" in node_feature_names:
            general_vertex_features.append(nv["normal_var_mean"])
        if "normal_var_max" in node_feature_names:
            general_vertex_features.append(nv["normal_var_max"])

    if "mean_curvature" in node_feature_names or "gaussian_curvature" in node_feature_names:
        curv = _mean_gaussian_curvature_on_boundary(wrapped_mesh)
        if "mean_curvature" in node_feature_names:
            general_vertex_features.append(curv["mean_curvature"])
        if "gaussian_curvature" in node_feature_names:
            general_vertex_features.append(curv["gaussian_curvature"])

    general_vertex_features_arr = np.array(general_vertex_features).T if general_vertex_features else None

    if feature_provider is not None:
        # mold，进入此处
        problem_vertex_features = feature_provider.get_vertex_features(wrapped_mesh=wrapped_mesh)
        vertex_features = safe_concatenate([general_vertex_features_arr, problem_vertex_features], axis=1)
    else:
        vertex_features = general_vertex_features_arr

    if vertex_features is None:
        return torch.zeros((wrapped_mesh.num_vertices, 0), dtype=torch.float32)

    return torch.tensor(vertex_features, dtype=torch.float32)


def get_inter_graph_edges(
    src_mesh: MeshWrapper, dest_mesh: MeshWrapper, node_type: str, edge_feature_names: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get symmetric edges from one graph to the other. The edge features are computed based on the edge_feature_names.
    Already includes index offsets for the second graph.
    """
    if node_type == "element":
        index_offset = src_mesh.num_elements
        src_node_indices = np.arange(src_mesh.num_elements)
        src_node_positions = src_mesh.element_midpoints
        dest_node_indices = dest_mesh.find_closest_elements(src_node_positions)
        dest_node_positions = dest_mesh.element_midpoints[dest_node_indices]
        dest_node_indices = dest_node_indices + index_offset

    elif node_type == "vertex":
        index_offset = src_mesh.num_vertices
        src_node_indices = np.arange(src_mesh.num_vertices)
        src_node_positions = src_mesh.vertex_positions
        dest_node_indices = dest_mesh.vertex_tree.query(src_node_positions, k=1)[1]
        dest_node_positions = dest_mesh.vertex_positions[dest_node_indices]
        dest_node_indices = dest_node_indices + index_offset

    else:
        raise ValueError(f"Node type {node_type=} not supported")

    positions1 = np.concatenate((src_node_positions, dest_node_positions), axis=0)
    positions2 = np.concatenate((dest_node_positions, src_node_positions), axis=0)

    indices1 = np.concatenate((src_node_indices, dest_node_indices), axis=0)
    indices2 = np.concatenate((dest_node_indices, src_node_indices), axis=0)

    # ====== Build edge features in EXACT order of edge_feature_names ======
    edge_features: List[np.ndarray] = []

    # 预先算好常用量，避免循环里重复算
    delta = positions1 - positions2  # (N, dim)
    euclid = np.linalg.norm(delta, axis=1).astype(np.float32)  # (N,)
    N = len(indices1)

    for fname in edge_feature_names:
        # --- vector-valued feature: expands into dim columns ---
        if fname == "distance_vector":
            edge_features.extend(list(delta.T.astype(np.float32)))  # dim columns

        # --- scalar distance ---
        elif fname == "euclidean_distance":
            edge_features.append(euclid)

        # --- features that are undefined / not meaningful for inter-graph edges ---
        # vertex inter edges: no true mesh edge, no boundary semantics, no sizing gradient along real edge, etc.
        elif fname in {
            "edge_curvature",
            "boundary_edge_flag",
            "edge_length_over_sqrt_hihj",
            "delta_log_sizing_field",
            "dihedral_angle",
            "sharp_edge_flag",
        }:
            edge_features.append(np.zeros(N, dtype=np.float32))

        else:
            raise ValueError(
                f"Unknown inter-edge feature name '{fname}'. "
                f"Please add it to get_inter_graph_edges or remove it from edge_feature_names."
            )

    edge_index = torch.tensor(np.vstack((indices1, indices2))).long()
    edge_features_arr = (
        np.array(edge_features).T if edge_features else np.zeros((len(indices1), 0), dtype=np.float32)
    )
    edge_attr = torch.tensor(edge_features_arr, dtype=torch.float32)

    return edge_attr, edge_index
