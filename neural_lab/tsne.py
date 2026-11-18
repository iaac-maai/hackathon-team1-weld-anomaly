"""t-SNE of the latent space, and clusters of the anomalies.        python -m neural_lab.tsne

1. Every test tile (3,019 tiles, 4 welds never used in training) goes through the published encoder:
   32 latent numbers per tile. Nothing is retrained; the weights are the published ones.
2. t-SNE squeezes those 32 numbers into 2 for a map (a picture only: distances between far-apart groups on a
   t-SNE map mean little, and its shape changes with perplexity).
3. The ANOMALIES are the tiles the model itself flags (score above t_low: CHECK or HIGH). They are clustered with
   k-means in the full 32-number latent space (not on the t-SNE picture). k is chosen by silhouette score;
   k = 3 (the number of defect types in RIAWELC) is also reported.
4. The clusters are made WITHOUT the labels. Only afterwards are they compared with the true defect types
   (crack, porosity, lack of penetration): adjusted Rand index and a cross-table. So the comparison is a test,
   not a fit.

Writes neural_lab/outputs/ (git-ignored, the example sheet contains RIAWELC pixels):
  tsne_map.png        t-SNE of all test tiles: true type | anomaly score | anomaly clusters
  tsne_examples.png   six example tiles per cluster
  tsne_clusters.csv   per tile: path, true type, score, risk, t-SNE x/y, cluster (-1 = not flagged)
  tsne_summary.json   the numbers printed below
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .data import CACHE_DIR, DEFECTS, ORDER
from .explorer import Explorer
from .verify import load_tiles

OUT = CACHE_DIR.parent / "outputs"
COL = {"no_defect": "#55A868", "porosity": "#4C72B0", "crack": "#C44E52", "lack_of_penetration": "#8172B2"}
SEED = 42


@torch.no_grad()
def latent_codes(model, X, batch=256) -> np.ndarray:
    return torch.cat([model.encoder(X[i:i + batch]) for i in range(0, len(X), batch)]).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perplexity", type=float, default=30)
    ap.add_argument("--kmax", type=int, default=6)
    a = ap.parse_args()

    ex = Explorer("cpu")
    test = ex.test.reset_index(drop=True)
    cache = CACHE_DIR / "test_latent.npy"
    if cache.exists() and len(np.load(cache)) == len(test):
        Z = np.load(cache)
    else:
        print(f"encoding {len(test)} test tiles ...")
        Z = latent_codes(ex.model, load_tiles(test["path"], ex.size))
        np.save(cache, Z)
    Zs = StandardScaler().fit_transform(Z)

    print(f"t-SNE (perplexity {a.perplexity}) ...")
    xy = TSNE(n_components=2, perplexity=a.perplexity, init="pca", random_state=SEED).fit_transform(Zs)
    test["tsne_x"], test["tsne_y"] = xy[:, 0], xy[:, 1]

    # anomalies = what the model flags (no labels used)
    flag = (test["risk"] != "low").to_numpy()
    Za = Zs[flag]
    sil = {k: float(silhouette_score(Za, KMeans(k, n_init=10, random_state=SEED).fit_predict(Za))) for k in range(2, a.kmax + 1)}
    k_best = max(sil, key=sil.get)
    runs = {}
    for k in sorted({k_best, 3}):
        lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Za)
        truth = test.loc[flag, "label"].to_numpy()
        runs[k] = {"labels": lab,
                   "ari_vs_true_type": float(adjusted_rand_score(truth, lab)),
                   "crosstab": pd.crosstab(pd.Series(lab, name="cluster"), pd.Series(truth, name="true type"))
                                 .reindex(columns=[c for c in ORDER if c in set(truth)], fill_value=0)}
    show = k_best
    test["cluster"] = -1
    test.loc[flag, "cluster"] = runs[show]["labels"]

    # ---------------- figures
    OUT.mkdir(exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for lab in ORDER:
        m = test["label"] == lab
        axs[0].scatter(xy[m, 0], xy[m, 1], s=5, c=COL[lab], label=f"{lab} ({m.sum()})", alpha=.7)
    axs[0].legend(markerscale=3, fontsize=8); axs[0].set_title("True type (not used to make the map)")
    sc = axs[1].scatter(xy[:, 0], xy[:, 1], s=5, c=np.log10(test["score"]), cmap="magma")
    fig.colorbar(sc, ax=axs[1], label="log10 anomaly score"); axs[1].set_title("Anomaly score")
    axs[2].scatter(xy[~flag, 0], xy[~flag, 1], s=4, c="#dddddd", label=f"not flagged ({(~flag).sum()})")
    cmap = plt.get_cmap("tab10")
    for c in range(show):
        m = (test["cluster"] == c).to_numpy()
        axs[2].scatter(xy[m, 0], xy[m, 1], s=6, color=cmap(c), label=f"cluster {c} ({m.sum()})")
    axs[2].legend(markerscale=3, fontsize=8)
    axs[2].set_title(f"Anomaly clusters (k-means, k={show}, in the 32-number latent space)")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"t-SNE of the latent space · {len(test):,} test tiles · perplexity {a.perplexity:g}")
    fig.tight_layout(); fig.savefig(OUT / "tsne_map.png", dpi=160); plt.close(fig)

    from .data import tile_path
    from PIL import Image
    fig, axs = plt.subplots(show, 6, figsize=(9, 1.6 * show), squeeze=False)
    rng = np.random.default_rng(SEED)
    for c in range(show):
        idx = test.index[test["cluster"] == c].to_numpy()
        for j, i in enumerate(rng.choice(idx, size=min(6, len(idx)), replace=False)):
            axs[c, j].imshow(Image.open(tile_path(test.loc[i, "path"])).convert("L"), cmap="gray")
            axs[c, j].set_title(f"{test.loc[i, 'label'][:12]}\n{test.loc[i, 'score']:.4f}", fontsize=6)
        for j in range(6):
            axs[c, j].axis("off")
        axs[c, 0].text(-0.15, .5, f"cluster {c}", transform=axs[c, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "tsne_examples.png", dpi=160); plt.close(fig)

    test[["path", "label", "score", "risk", "tsne_x", "tsne_y", "cluster"]].to_csv(OUT / "tsne_clusters.csv", index_label="test_index")
    summary = {"n_test": len(test), "n_flagged": int(flag.sum()),
               "flagged_by_true_type": test.loc[flag, "label"].value_counts().reindex(ORDER, fill_value=0).to_dict(),
               "perplexity": a.perplexity, "silhouette_by_k": sil, "k_best": k_best,
               "runs": {k: {"ari_vs_true_type": r["ari_vs_true_type"], "crosstab": {int(i): {c: int(v) for c, v in row.items()} for i, row in r["crosstab"].to_dict(orient="index").items()}}
                        for k, r in runs.items()}}
    (OUT / "tsne_summary.json").write_text(json.dumps(summary, indent=1, default=int))

    print(f"\nflagged as anomalies (CHECK or HIGH): {flag.sum()} of {len(test)}  {summary['flagged_by_true_type']}")
    print("silhouette by k:", {k: round(v, 3) for k, v in sil.items()}, "-> best k =", k_best)
    for k, r in runs.items():
        print(f"\nk = {k}: agreement with the true defect type (adjusted Rand index, 0 = chance, 1 = identical): "
              f"{r['ari_vs_true_type']:.3f}")
        print(r["crosstab"].to_string())
    print(f"\nwrote {OUT / 'tsne_map.png'}, tsne_examples.png, tsne_clusters.csv, tsne_summary.json")


if __name__ == "__main__":
    main()
