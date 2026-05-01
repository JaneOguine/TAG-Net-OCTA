import random
import numpy as np
import torch
import torch.nn.functional as F
from util.utils import *
from monai.losses import *
from skimage.morphology import medial_axis
from scipy.ndimage import convolve


def convert_to_graph(skel):
    """
    skel: binary skeleton (numpy array, 0/1)
    """
    skel = skel.astype(np.uint8)

    kernel = np.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=np.uint8)

    neighbor_count = convolve(skel, kernel, mode="constant", cval=0)

    endpoints = (skel == 1) & (neighbor_count == 1)
    junctions = (skel == 1) & (neighbor_count >= 3)

    return {
        "neighbor_count": neighbor_count,
        "endpoints": endpoints,
        "junctions": junctions,
    }


def extract_edges(skel, endpoints, junctions):
    skel = skel.copy()
    visited = np.zeros_like(skel, dtype=np.uint8)

    nodes = np.argwhere(endpoints | junctions)
    node_set = set(map(tuple, nodes.tolist()))

    edges = []

    for node_arr in nodes:
        node = tuple(node_arr.tolist())

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                x, y = node[0] + dx, node[1] + dy

                if not (0 <= x < skel.shape[0] and 0 <= y < skel.shape[1]):
                    continue

                if skel[x, y] == 0 or visited[x, y]:
                    continue

                edge = [node]
                cx, cy = x, y

                while True:
                    edge.append((cx, cy))
                    visited[cx, cy] = 1

                    if (cx, cy) in node_set and (cx, cy) != node:
                        break

                    found_next = False
                    for dx2 in [-1, 0, 1]:
                        for dy2 in [-1, 0, 1]:
                            if dx2 == 0 and dy2 == 0:
                                continue

                            nx, ny = cx + dx2, cy + dy2

                            if not (0 <= nx < skel.shape[0] and 0 <= ny < skel.shape[1]):
                                continue

                            if skel[nx, ny] == 1 and not visited[nx, ny]:
                                cx, cy = nx, ny
                                found_next = True
                                break
                        if found_next:
                            break

                    if not found_next:
                        break

                if len(edge) > 1:
                    edges.append(edge)

    return edges


def get_graph_info(skel):
    """
    skel: torch tensor [H, W] or numpy array [H, W]
    """
    if isinstance(skel, torch.Tensor):
        skel = skel.detach().cpu().numpy()

    skel = skel.astype(np.uint8)

    graph_info = convert_to_graph(skel)
    edges = extract_edges(skel, graph_info["endpoints"], graph_info["junctions"])

    graph = {
        "nodes": np.argwhere(graph_info["endpoints"] | graph_info["junctions"]),
        "edges": edges,
    }
    return graph


def coord_to_index(x, y, W):
    return x * W + y


def sample_edge_pixels(edge, max_samples=20):
    if len(edge) <= max_samples:
        return edge
    return random.sample(edge, max_samples)


def edge_centroid(edge, device):
    return torch.tensor(edge, device=device, dtype=torch.float32).mean(dim=0)


def graph_contrastive_loss_single(
    embeddings,
    graph,
    W,
    seg_pred=None,
    max_pos_samples=5,          # reduced for speed
    max_neg_samples=20,
    temperature=0.1,
    num_pos_per_anchor=3,
):
    """
    embeddings: [C, H, W]
    graph: dict with "edges"
    seg_pred: [H, W] or None
    """

    device = embeddings.device
    C, H, W = embeddings.shape

    emb = embeddings.permute(1, 2, 0).reshape(-1, C)  # [N, C]
    loss = torch.tensor(0.0, device=device)
    count = 0

    edges = graph["edges"]
    if len(edges) < 2:
        return loss

    # ---------- Precompute ----------
    edge_indices = [
        torch.tensor(
            [coord_to_index(x, y, W) for x, y in edge],
            device=device,
            dtype=torch.long
        )
        for edge in edges
    ]

    centroids = [
        torch.tensor(edge, device=device, dtype=torch.float32).mean(dim=0)
        for edge in edges
    ]

    seg_flat = seg_pred.view(-1) if seg_pred is not None else None

    # ---------- Main loop ----------
    for i in range(len(edges)):

        if len(edge_indices[i]) < 5:
            continue

        c_i = centroids[i]

        # compute distances to all other edges (vectorized)
        dist_list = []
        idx_list = []
        for j, c_j in enumerate(centroids):
            if j == i:
                continue
            dist_list.append(torch.norm(c_i - c_j))
            idx_list.append(j)

        if len(dist_list) == 0:
            continue

        dists = torch.stack(dist_list)
        neg_edge_idx = idx_list[torch.argmin(dists).item()]

        pos_candidates = edge_indices[i]
        neg_candidates = edge_indices[neg_edge_idx]

        if len(neg_candidates) == 0:
            continue

        # sample anchors
        perm = torch.randperm(len(pos_candidates), device=device)
        anchor_indices = pos_candidates[perm[:max_pos_samples]]

        for a_idx in anchor_indices:

            if seg_flat is not None and seg_flat[a_idx] == 0:
                continue

            anchor_emb = emb[a_idx]

            # sample positives
            perm_p = torch.randperm(len(pos_candidates), device=device)
            pos_idx = pos_candidates[perm_p[:num_pos_per_anchor]]

            # sample negatives
            perm_n = torch.randperm(len(neg_candidates), device=device)
            neg_idx = neg_candidates[perm_n[:max_neg_samples]]

            # -------- filtering --------
            if seg_flat is not None:
                pos_mask = seg_flat[pos_idx] != 0
                neg_mask = seg_flat[neg_idx] != 0

                pos_idx = pos_idx[pos_mask]
                neg_idx = neg_idx[neg_mask]

            if pos_idx.numel() == 0 or neg_idx.numel() == 0:
                continue

            pos_embs = emb[pos_idx]  # [P, C]
            neg_embs = emb[neg_idx]  # [K, C]

            # -------- similarity --------
            pos_sim = F.cosine_similarity(
                anchor_emb.unsqueeze(0), pos_embs, dim=1
            )
            neg_sim = F.cosine_similarity(
                anchor_emb.unsqueeze(0), neg_embs, dim=1
            )

            # -------- optional weighting --------
            weight = 1.0
            if seg_flat is not None:
                anchor_class = seg_flat[a_idx]
                neg_classes = seg_flat[neg_idx]
                if torch.any(neg_classes != anchor_class):
                    weight = 1.5

            # -------- InfoNCE --------
            pos_exp = torch.exp(pos_sim / temperature)
            neg_exp = torch.exp(neg_sim / temperature)

            numerator = pos_exp.sum()
            denominator = numerator + neg_exp.sum()

            loss += weight * (-torch.log(numerator / (denominator + 1e-8)))
            count += 1

    if count == 0:
        return torch.tensor(0.0, device=device)

    return loss / count

def graph_contrastive_loss_batch(
    embeddings,
    graphs,
    seg_preds,
    max_pos_samples=6,
    max_neg_samples=20,
    temperature=0.1,
    num_pos_per_anchor=3,
):
    """
    embeddings: [B, C, H, W]
    graphs: list of graph dicts, len=B
    seg_preds: [B, H, W]
    """
    B, C, H, W = embeddings.shape
    total_loss = torch.tensor(0.0, device=embeddings.device)

    for b in range(B):
        emb_b = embeddings[b]
        graph_b = graphs[b]
        seg_b = seg_preds[b]

        loss_b = graph_contrastive_loss_single(
            embeddings=emb_b,
            graph=graph_b,
            W=W,
            seg_pred=seg_b,
            max_pos_samples=max_pos_samples,
            max_neg_samples=max_neg_samples,
            temperature=temperature,
            num_pos_per_anchor=num_pos_per_anchor,
        )
        total_loss = total_loss + loss_b

    return total_loss / B