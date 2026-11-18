"""Neural Network Lab: local Gradio app to look inside the team's pretrained weld autoencoder.

    python -m neural_lab.app        (from the repository root; opens http://127.0.0.1:7861)

Local only and separate from 02_model_server/ (the deployed app is not touched or imported).
Every number and image shown comes from a real forward pass of the published model.
"""
import os

import gradio as gr

from . import plots
from .activations import BLOCKS
from .data import DEFECTS, ORDER
from .explorer import Explorer

EX = Explorer("cpu")      # CPU: same numerics as the deployed ONNX server; one 64x64 tile takes milliseconds
_ = EX.test               # score the test welds once at start-up (cached on disk afterwards)

ABOUT = """
### What you are looking at
- **Input → encoder → latent code → decoder → reconstruction.** The model only ever saw *sound* welds in
  training, so it rebuilds sound tiles well and defects badly. The **error map** (input − reconstruction)² shows where.
- **Feature maps** are the outputs of the three encoder blocks (Conv 3×3 stride 2 → BatchNorm → ReLU), recorded
  with PyTorch *forward hooks* during the real forward pass. Bright = the channel responded strongly there.
  "Strongest" only *ranks* channels by their mean activation for this tile: it does not say what a channel means.
- **Latent code**: the 32 numbers the encoder squeezes each tile into. The decoder rebuilds the tile from these alone.
- **Tiles** come from the team's **test welds** (RRT-23R, RRT-28R, RRT-31R, RRT-40R), never used for training.
  Preprocessing, weights, score and risk thresholds are exactly those of the deployed model.
- RIAWELC images: Totino, Spagnolo, Perri (ICMECE 2022) and Perri et al. (Manufacturing Letters). Not redistributed.
"""


def _summary(row, inf, risk):
    where = f"test weld **{row['weld']}**, `{row['source']}`, true class **{row['label']}**" if row is not None else "uploaded image"
    return (f"**Score** {inf.score:.6f} → **{risk.upper()}**  (low ≤ {EX.config['t_low']:.6f} < check ≤ "
            f"{EX.config['t_high']:.6f} < high) · {where}")


def explore(source, label, which, rank, upload, k, order):
    if source == "upload":
        if upload is None:
            raise gr.Error("Upload a weld X-ray tile first.")
        row, (inf, risk) = None, EX.infer_image(upload)
    else:
        row = EX.pick(label, which, rank)
        inf, risk = EX.infer_tile(row)
    title = "uploaded tile" if row is None else f"{label} ({which})"
    return (_summary(row, inf, risk), plots.overview(inf, risk, title),
            *[plots.feature_maps(inf, b, k, order) for b in BLOCKS], plots.latent([(title, inf)]))


def compare(defect, which):
    rows = [EX.pick("no_defect", which), EX.pick(defect, which)]
    items = [(r["label"], *EX.infer_tile(r)) for r in rows]
    table = "| tile | weld | score | risk |\n|---|---|---|---|\n" + "\n".join(
        f"| {r['label']} | {r['weld']} | {inf.score:.6f} | {risk} |" for r, (_, inf, risk) in zip(rows, items))
    return table, plots.compare(items), plots.latent([(n, inf) for n, inf, _ in items])


with gr.Blocks(title="Weld AE · Neural Network Lab") as demo:
    gr.Markdown("# Neural Network Lab · inside the weld-anomaly autoencoder\n"
                "Team 1 pretrained `ConvAE(64, 32)` from Hugging Face, loaded without retraining.")
    with gr.Tab("Explore one tile"):
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(["test tile", "upload"], value="test tile", label="Image source")
                label = gr.Dropdown(ORDER, value="porosity", label="Class (test welds)")
                which = gr.Radio(["median", "lowest", "highest", "rank"], value="median",
                                 label="Which tile of that class (by its real score)")
                rank = gr.Slider(0, 950, value=0, step=1, label="Rank (0 = lowest score), used when 'rank' is chosen")
                upload = gr.Image(type="pil", label="Or upload a RIAWELC-style tile", image_mode="L")
                k = gr.Slider(4, 32, value=16, step=4, label="Channels to show per block")
                order = gr.Radio(["strongest", "index"], value="strongest", label="Channel order")
                run_btn = gr.Button("Run the model", variant="primary")
            with gr.Column(scale=3):
                summary = gr.Markdown()
                overview = gr.Plot(label="Input, reconstruction, error")
                block_plots = [gr.Plot(label=plots.BLOCK_INFO[b]) for b in BLOCKS]
                latent_plot = gr.Plot(label="Latent code")
        # Uploading an image switches the source to "upload" (otherwise the upload would be ignored)
        upload.upload(lambda img: "upload", upload, source)
        run_btn.click(explore, [source, label, which, rank, upload, k, order],
                      [summary, overview, *block_plots, latent_plot])
    with gr.Tab("Sound vs defect"):
        with gr.Row():
            defect = gr.Dropdown(DEFECTS, value="crack", label="Defect class")
            which_c = gr.Radio(["median", "lowest", "highest"], value="median", label="Tile of each class")
            cmp_btn = gr.Button("Compare", variant="primary")
        cmp_table = gr.Markdown()
        cmp_plot = gr.Plot(label="Same model, sound vs defect")
        cmp_latent = gr.Plot(label="Latent codes")
        cmp_btn.click(compare, [defect, which_c], [cmp_table, cmp_plot, cmp_latent])
    with gr.Tab("About"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    # Port 7861 by default; another one with:  $env:GRADIO_SERVER_PORT="7862"
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7861)), inbrowser=True)
