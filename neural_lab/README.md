# Neural Network Lab · inside the weld-anomaly autoencoder

An isolated, local experiment by Sahil on top of Team 1's project. It loads the **published, pretrained**
`ConvAE(64, 32)` from Hugging Face ([iaac-maai/hackathon-team1-weld-anomaly-ae](https://huggingface.co/iaac-maai/hackathon-team1-weld-anomaly-ae)),
runs real tiles through it and shows what happens inside. Nothing is retrained. The deployed app
(`02_model_server/`) is neither changed nor imported.

## Run it

From the repository root, in the `iaac-ai` environment:

```powershell
python -m neural_lab.data                  # once: get + extract RIAWELC, check counts, rebuild the team split
python -m neural_lab.verify                # PyTorch == ONNX, published threshold and ROC AUC reproduced
python -m pytest neural_lab/tests -q       # tests
python -m neural_lab.app                   # the explorer: http://127.0.0.1:7861
python -m neural_lab.render_examples       # optional: save the figures to neural_lab/outputs/
python -m neural_lab                       # NEURAL / LAB: the main app (http://127.0.0.1:7870)
```

### NEURAL / LAB — Exploring the Black Box

`python -m neural_lab`, then http://127.0.0.1:7870. The published model runs **in the browser** (ONNX Runtime Web).
Three rooms share one weld (pick a test tile or upload one):

- **Defect experiment** (`#defect`): draw on a large canvas; live score, LOW / CHECK / HIGH, heatmap overlay,
  before/after wipe, and an original → redraw → error sequence (crossfades between real images only). Below, the
  sensitivity study: one synthetic pore / crack / band on 12 sound test welds, swept in size or darkness, against
  real defect scores; CSV export.
- **Architecture** (`#architecture`): the real ConvAE as nine stations; planes are real feature maps of encoder and
  decoder, bars are real per-channel means; click a station for all its channels; drag the 32 latent bars and the
  decoder answers. Lines show the order of operations, not weights.
- **Latent universe** (`#universe`): the 3,019 test tiles by PCA of their codes (3D PC1–PC3 or 2D), true-class
  colours, hover for the real tile and score, click to load it; your weld is projected live.
- **Play 90-second demo**: a guided tour that drives the real model; captions about results are written from the
  measured result. Esc or "Stop demo" ends it.
- **Is it a neural network?** (`#proof`): the evidence. Every layer read from the published weight file (718,369
  learned values), the trained network vs its untrained twin (same architecture, random weights) live on your weld
  and as ROC AUC on the 3,019 test tiles against a no-network contrast baseline, the 32 learned first-layer filters,
  and the training record. Built once by `python -m neural_lab.playground.proof` (writes `static/proof.json`,
  `static/control.onnx`, `static/training_curves.png`: no dataset pixels).
- `/classic.html`: the previous single-page interface, kept as a fallback.

`playground/export.py` prepares the browser model: encoder (with its three block outputs) and decoder (with its three
intermediate outputs) exported from the published weights and checked against the deployed ONNX model, PyTorch and
forward hooks; it also measures neuron influence and computes the PCA. The page lists these checks.

## Files

| File | What it does |
|---|---|
| `model.py` | `ConvAE` copied unchanged from the team notebook; loads the published weights (`strict=True`); the exact preprocessing (greyscale, 64×64 bilinear, ÷255) |
| `activations.py` | forward hooks that record the 3 encoder blocks and the 32-number latent code during one real forward pass |
| `data.py` | downloads/extracts RIAWELC (19 RAR parts), checks the counts, rebuilds the team's split by weld and checks it against the notebook |
| `verify.py` | the model is the deployed one: same ONNX file, same outputs, same threshold and ROC AUC |
| `plots.py` | figures: input / reconstruction / error map, feature maps, latent code, sound-vs-defect comparison |
| `explorer.py`, `app.py` | the local Gradio interface |
| `playground/` | NEURAL / LAB: export + checks, a small local server (127.0.0.1 only), the in-browser page |
| `tests/` | pytest checks for all of the above |

Git-ignored: `neural_lab/.cache/` (split table, cached test scores) and `neural_lab/outputs/` (figures contain
RIAWELC pixels, which must not be redistributed). The weights stay in the Hugging Face cache, outside the repo.

## The ideas, in plain words

- **Neural network.** A big function made of simple steps (layers). Each step multiplies its input by learned
  numbers (weights) and passes the result on. Training adjusted those weights once; here we only *use* them.
- **Autoencoder.** Two halves. The **encoder** squeezes a 64×64 tile (4,096 pixels) into **32 numbers**; the
  **decoder** redraws the tile from those 32 numbers alone. Ours learned only from sound welds.
- **Convolution and feature maps.** A convolution slides a small 3×3 filter over the image and asks "does this
  little pattern appear here?" at every position. Each filter produces one **feature map**: a picture of where
  it responded. A block has many filters (32, 64, 128 **channels**), so many feature maps.
- **Stride 2.** The filter jumps 2 pixels at a time, so each block halves the size: 64 → 32 → 16 → 8 px. Deeper
  maps are smaller but each pixel "sees" a bigger area of the tile.
- **BatchNorm and ReLU.** BatchNorm rescales values to a stable range; ReLU keeps positives and sets negatives to
  0. That is why the feature maps are never negative and often have dark (zero) regions.
- **Activation.** The value a layer outputs for a given input. "Strong activation" = large value. We do not claim
  what a channel *means*: "strongest" only ranks channels by their average value for that tile.
- **Forward hook.** A small function PyTorch calls right after a layer runs. We use it to copy the layer's output
  without changing the model. The tests check that the output is identical with and without hooks.
- **Latent code.** The 32 numbers in the middle. Everything the decoder knows about the tile is in there.
- **Reconstruction error.** (input − reconstruction)² per pixel = the **error map**; its mean = the **anomaly
  score**. The model never learned to draw pores or cracks, so it fails exactly there and the score rises.
- **ONNX.** A portable copy of the model used by the deployed server. `verify.py` checks it gives the same answers.

## Data and citation

RIAWELC (no formal licence, free with citation, not redistributed): B. Totino, F. Spagnolo, S. Perri, "RIAWELC:
A Novel Dataset of Radiographic Images for Automatic Weld Defects Classification", ICMECE 2022; S. Perri,
F. Spagnolo, F. Frustaci, P. Corsonello, "Welding Defects Classification Through a Convolutional Neural Network",
Manufacturing Letters.
