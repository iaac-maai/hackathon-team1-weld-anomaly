"""Evidence that NEURAL / LAB runs a trained neural network, for the "Proof" room.     From the repository root:

    python -m neural_lab.playground.proof

Writes two files into neural_lab/playground/static/ (small, no dataset pixels, safe to commit and to serve online):
- proof.json    the network read from the published weight file: every layer, its weight shapes and number of learned
                values (total 718,369), the 32 first-layer filters (trained vs untrained), the computation per weld,
                and a control experiment on the 3,019 test tiles: trained network vs the SAME architecture with
                untrained (random) weights vs a no-network baseline (pixel contrast), all scored with ROC AUC
- control.onnx  the untrained twin (same architecture, random weights, seed 0), so the page can run it live
Training facts are copied from the team notebook's printed output (SECTION 8) and cited as such.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn

from ..data import evaluation_sets, load_split, tile_path
from ..model import ConvAE, REPO_ROOT, hf_file, load_pretrained, n_params
from ..verify import load_tiles, scores_torch

STATIC = Path(__file__).resolve().parent / "static"
CONTROL_SEED = 0
TRAINING = {   # team notebook, SECTION 8 output for the published setup D (ConvAE, 64 px, latent 32)
    "optimizer": "Adam, learning rate 0.001", "loss": "mean squared error between input and reconstruction",
    "batch": 64, "data": "3,783 sound tiles of the 20 training welds, random flips",
    "best_epoch": 21, "epochs_run": 27, "early_stopping": "stop after 6 epochs without improvement on sound validation tiles",
    "val_mse": 0.00020, "seconds": 8.2, "hardware": "RTX 5080 Laptop GPU",
    "source": "01_autoencoder/notebook/autoencoder.ipynb, SECTION 8 (printed: 'setup D: best epoch 21 of 27, val MSE 0.00020, 8.2 s')",
}


def layer_table(model: nn.Module, size: int = 64):
    """Every layer with learned values, in execution order, plus how many multiply-adds it does per weld."""
    rows, hooks, shapes = [], [], {}
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(lambda mod, i, o, n=name: shapes.__setitem__(n, (tuple(i[0].shape), tuple(o.shape)))))
    with torch.no_grad():
        model(torch.zeros(1, 1, size, size))
    for h in hooks:
        h.remove()
    for name, m in model.named_modules():
        if len(list(m.children())) or name not in shapes:
            continue
        params = {k: list(p.shape) for k, p in m.named_parameters(recurse=False)}
        n = sum(p.numel() for p in m.parameters(recurse=False))
        (inp, out) = shapes[name]
        macs = 0
        if isinstance(m, nn.Conv2d):
            macs = m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * out[2] * out[3]
        elif isinstance(m, nn.ConvTranspose2d):
            macs = m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] * inp[2] * inp[3]
        elif isinstance(m, nn.Linear):
            macs = m.in_features * m.out_features
        rows.append({"layer": name.replace(".", " "), "type": type(m).__name__, "weights": params, "learned": n,
                     "input": list(inp[1:]), "output": list(out[1:]), "multiply_adds": macs})
    return rows


@torch.no_grad()
def main():
    model, config = load_pretrained("cpu")
    torch.manual_seed(CONTROL_SEED)
    control = ConvAE(config["image_size"], config["latent_dim"]).eval()   # same architecture, never trained

    layers = layer_table(model)
    total = sum(r["learned"] for r in layers)
    assert total == n_params(model) == 718_369, total

    # control experiment on the team's test welds (inference only)
    _, _, test = evaluation_sets(load_split())
    X = load_tiles(test["path"])
    y = (test["label"] != "no_defect").to_numpy().astype(int)
    s_trained, s_control = scores_torch(model, X), scores_torch(control, X)
    contrast = np.array([np.asarray(Image.open(tile_path(p)), dtype=np.float32).std() for p in test["path"]])
    sound = y == 0
    control_exp = {
        "n_test": int(len(y)), "n_sound": int(sound.sum()), "n_defect": int((~sound).sum()),
        "auc_trained": float(roc_auc_score(y, s_trained)),
        "auc_untrained": float(roc_auc_score(y, s_control)),
        "auc_contrast_baseline": float(roc_auc_score(y, contrast)),
        "mse_sound_trained": float(s_trained[sound].mean()), "mse_sound_untrained": float(s_control[sound].mean()),
        "rule": "ROC AUC: chance that a random defect tile scores higher than a random sound tile (1 = perfect, 0.5 = coin flip)",
    }

    k_trained = model.encoder[0].weight[:, 0].numpy()        # (32, 3, 3)
    k_control = control.encoder[0].weight[:, 0].numpy()
    proof = {
        "weights_file": "anomaly_ae.pt (Hugging Face iaac-maai/hackathon-team1-weld-anomaly-ae)",
        "weights_sha256": hashlib.sha256(hf_file("anomaly_ae.pt").read_bytes()).hexdigest(),
        "total_learned": total, "layers": layers,
        "multiply_adds_per_weld": int(sum(r["multiply_adds"] for r in layers)),
        "filters_trained": np.round(k_trained, 4).tolist(), "filters_untrained": np.round(k_control, 4).tolist(),
        "control_seed": CONTROL_SEED, "training": TRAINING, "control": control_exp,
    }
    (STATIC / "proof.json").write_text(json.dumps(proof))

    torch.onnx.export(control, (torch.zeros(1, 1, 64, 64),), STATIC / "control.onnx", input_names=["image"],
                      output_names=["reconstruction"], opset_version=18, dynamo=False)
    fig = REPO_ROOT / "01_autoencoder" / "figures" / "training_curves.png"
    if fig.exists():
        (STATIC / "training_curves.png").write_bytes(fig.read_bytes())
    print(f"layers {len(layers)}, learned values {total:,}, multiply-adds per weld {proof['multiply_adds_per_weld']:,}")
    print(f"ROC AUC on {len(y)} test tiles: trained {control_exp['auc_trained']:.3f} | untrained twin "
          f"{control_exp['auc_untrained']:.3f} | contrast baseline {control_exp['auc_contrast_baseline']:.3f}")
    print(f"wrote {STATIC / 'proof.json'}, control.onnx" + (", training_curves.png" if fig.exists() else ""))


if __name__ == "__main__":
    main()
