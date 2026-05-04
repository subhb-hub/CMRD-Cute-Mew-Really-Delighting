import argparse
import math
from pathlib import Path


DEFAULT_CHANNEL_ORDER = "data/SEED-IV-RAW/Channel Order.xlsx"
DEFAULT_LOCS = "data/SEED-IV-RAW/channel_62_pos.locs"
DEFAULT_SAVE_ROOT = "data/SEED-IV"

np = None
pd = None
sparse = None
cKDTree = None


def load_dependencies():
    global np, pd, sparse, cKDTree

    import numpy as _np
    import pandas as _pd
    from scipy import sparse as _sparse
    from scipy.spatial import cKDTree as _cKDTree

    np = _np
    pd = _pd
    sparse = _sparse
    cKDTree = _cKDTree


def norm_name(value):
    return str(value).strip().upper()


def load_channel_order(xlsx_path):
    df = pd.read_excel(xlsx_path, header=None)
    names = [norm_name(x) for x in df.iloc[:, 0].astype(str).tolist()]
    return [x for x in names if x not in ("CHANNEL_NAME", "", "NAN")]


def load_locs_polar(locs_path):
    mapping = {}
    with open(locs_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            name = norm_name(parts[-1])
            angle_deg = float(parts[-3])
            radius = float(parts[-2])
            mapping[name] = (angle_deg, radius)
    return mapping


def polar_to_xy(angle_deg, radius):
    angle = math.radians(angle_deg)
    return radius * math.sin(angle), radius * math.cos(angle)


def build_coords_from_locs(channel_names, locs_map):
    coords = np.zeros((len(channel_names), 2), dtype=np.float32)
    valid = np.zeros((len(channel_names),), dtype=bool)
    missing = []

    for i, name in enumerate(channel_names):
        if name in locs_map:
            angle_deg, radius = locs_map[name]
            coords[i] = polar_to_xy(angle_deg, radius)
            valid[i] = True
        else:
            coords[i] = [np.nan, np.nan]
            missing.append(name)

    if missing:
        print(f"[WARN] Missing channel coordinates: {missing}")

    if valid.any():
        valid_coords = coords[valid]
        center = valid_coords.mean(axis=0, keepdims=True)
        coords = coords - center
        radius_max = np.linalg.norm(valid_coords - center, axis=1).max()
        coords = coords / (radius_max if radius_max > 0 else 1.0)

    return coords, valid


def build_spatial_adjacency(coords, valid_mask, sigma=None, topk=6, thresh=0.0, self_loop=False):
    count = coords.shape[0]
    xy = coords.copy()
    xy[~valid_mask] = 1e6

    diff = xy[:, None, :] - xy[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    if sigma is None:
        nonzero = distances[(distances > 0) & (distances < 1e5)]
        sigma = float(np.median(nonzero)) if nonzero.size else 0.3

    adjacency = np.exp(-(distances ** 2) / (2 * sigma ** 2)).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0 if self_loop else 0.0)

    if thresh > 0:
        adjacency[distances > thresh] = 0.0

    if topk > 0:
        for i in range(count):
            if not valid_mask[i]:
                adjacency[i, :] = 0.0
                continue

            row = adjacency[i]
            row[i] = 0.0
            ranked = np.argsort(row)[::-1]
            keep = []

            for j in ranked:
                if valid_mask[j]:
                    keep.append(j)
                if len(keep) >= topk:
                    break

            mask = np.ones_like(row, dtype=bool)
            mask[keep] = False
            row[mask] = 0.0
            adjacency[i] = row

    adjacency[~valid_mask, :] = 0.0
    adjacency[:, ~valid_mask] = 0.0
    return adjacency, sigma


def build_idw_mapping(coords, valid_mask, grid_size=32, k_neighbors=4, eps=1e-6):
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        raise ValueError("No valid channel coordinates were found.")

    count = coords.shape[0]
    grid_h = grid_w = int(grid_size)
    gx = np.linspace(-1, 1, grid_w)
    gy = np.linspace(-1, 1, grid_h)
    grid_x, grid_y = np.meshgrid(gx, gy)
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    mask = (grid_points ** 2).sum(axis=1) <= 1.0
    mask_2d = mask.reshape(grid_h, grid_w)

    valid_coords = coords[valid_mask]
    tree = cKDTree(valid_coords)
    k = min(int(k_neighbors), valid_coords.shape[0])
    dists, idxs = tree.query(grid_points, k=k)

    rows, cols, vals = [], [], []
    for point_id in range(grid_points.shape[0]):
        if not mask[point_id]:
            continue

        point_dist = np.atleast_1d(dists[point_id])
        point_idx = np.atleast_1d(idxs[point_id])

        if np.min(point_dist) < 1e-12:
            weights = np.zeros_like(point_dist)
            weights[np.argmin(point_dist)] = 1.0
        else:
            weights = 1.0 / (point_dist + eps)
            weights = weights / weights.sum()

        rows.extend([point_id] * len(point_idx))
        cols.extend(valid_idx[point_idx].tolist())
        vals.extend(weights.tolist())

    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(grid_h * grid_w, count), dtype=np.float32)
    meta = {
        "grid_size": np.int32(grid_size),
        "grid_x": grid_x.astype(np.float32),
        "grid_y": grid_y.astype(np.float32),
        "mask": mask_2d.astype(np.bool_),
    }
    return matrix, meta


def save_outputs(args, channel_names, coords, valid, adjacency, sigma):
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    np.save(save_root / "A_spatial.npy", adjacency)
    np.savez(
        save_root / "A_meta.npz",
        coords=coords.astype(np.float32),
        valid=valid.astype(np.bool_),
        ch_names=np.array(channel_names, dtype=object),
        sigma_used=np.float32(sigma),
        adj_topk=np.int32(args.adj_topk),
        adj_thresh=np.float32(args.adj_thresh),
        self_loop=np.int32(1 if args.self_loop else 0),
    )

    if args.save_topo:
        matrix, meta = build_idw_mapping(coords, valid, grid_size=args.grid_size, k_neighbors=args.idw_k)
        sparse.save_npz(save_root / "W_topo_csr.npz", matrix)
        np.savez(
            save_root / "topo_mapping.npz",
            ch_names=np.array(channel_names, dtype=object),
            coords=coords.astype(np.float32),
            valid=valid.astype(np.bool_),
            grid_size=meta["grid_size"],
            grid_x=meta["grid_x"],
            grid_y=meta["grid_y"],
            mask=meta["mask"],
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Build a channel spatial adjacency matrix from an EEG .locs file.")
    parser.add_argument("--channel-order", type=str, default=DEFAULT_CHANNEL_ORDER)
    parser.add_argument("--locs", type=str, default=DEFAULT_LOCS)
    parser.add_argument("--save-root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--idw-k", type=int, default=4)
    parser.add_argument("--adj-topk", type=int, default=6)
    parser.add_argument("--adj-sigma", type=float, default=None)
    parser.add_argument("--adj-thresh", type=float, default=0.0)
    parser.add_argument("--self-loop", action="store_true")
    parser.add_argument("--save-topo", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    args = build_parser().parse_args()
    load_dependencies()

    channel_names = load_channel_order(args.channel_order)
    locs_map = load_locs_polar(args.locs)
    coords, valid = build_coords_from_locs(channel_names, locs_map)

    adjacency, sigma = build_spatial_adjacency(
        coords,
        valid,
        sigma=args.adj_sigma,
        topk=args.adj_topk,
        thresh=args.adj_thresh,
        self_loop=args.self_loop,
    )

    save_outputs(args, channel_names, coords, valid, adjacency, sigma)
    print(f"Saved spatial graph to {Path(args.save_root) / 'A_spatial.npy'}")


if __name__ == "__main__":
    main()
