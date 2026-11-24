"""Amazon-style "Customers say" aspect highlighter UI."""

import html
import os

import gradio as gr

from infer import DEFAULT_MODEL_PATH, highlight_reviews

MODEL_PATH = os.environ.get("ABSA_MODEL_PATH", DEFAULT_MODEL_PATH)
COLORS = {"POS": "#c8f2d0", "NEG": "#ffd1d1", "NEU": "#e2e8f0"}
NAMES = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}

EXAMPLE_REVIEWS = """Battery life is amazing, I get almost two full days on a single charge.
The screen is gorgeous but the battery drains way too fast when gaming.
Sound quality is crisp and the bass is punchy. Bluetooth pairing was painless.
Arrived two days late and the box was crushed, but the headphones themselves work fine.
Comfortable to wear for hours, though the ear cushions get warm after a while.
Customer service was useless and never answered my emails about the warranty.
Great value for the price. Build quality feels premium and the charging case is sturdy.
The app is clunky and the noise cancelling is only average."""


def render_review(review: dict) -> str:
    text, pos, parts = review["text"], 0, []
    for s in sorted(review["spans"], key=lambda s: s["start"]):
        tip = f'{NAMES[s["sentiment"]]} · {s["score"]:.0%} confidence'
        parts.append(html.escape(text[pos : s["start"]]))
        parts.append(
            f'<mark style="background:{COLORS[s["sentiment"]]};padding:0 3px;border-radius:4px" '
            f'title="{html.escape(tip)}">{html.escape(s["text"])}</mark>'
        )
        pos = s["end"]
    parts.append(html.escape(text[pos:]))
    return f'<div style="padding:10px 0;border-bottom:1px solid #e5e7eb;line-height:1.7">{"".join(parts)}</div>'


def render(result: dict | None, aspect: str) -> str:
    if not result:
        return "<p>Paste reviews and click <b>Analyze Reviews</b>.</p>"
    chip = next((c for c in result["chips"] if c["aspect"] == aspect), None)
    reviews = [r for r in result["reviews"] if r["text"].strip()]
    if chip:
        reviews = [r for r in reviews if r["id"] in chip["review_ids"]]
    return "".join(render_review(r) for r in reviews) or "<p>No reviews to show.</p>"


def analyze(raw: str):
    try:
        result = highlight_reviews([line for line in raw.splitlines() if line.strip()], model_path=MODEL_PATH)
    except (OSError, ValueError) as exc:  # missing folder, unknown repo id, or no network for a first download
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        msg = (f"<p>⚠️ Could not load the model from <code>{html.escape(MODEL_PATH)}</code>: "
               f"{html.escape(reason)}<br>See <b>Quick start</b> in the README.</p>")
        return None, gr.update(choices=["All"], value="All"), msg
    choices = ["All"] + [
        (f'{c["aspect"]} ({c["pos"]}+ / {c["neg"]}-)', c["aspect"]) for c in result["chips"]
    ]
    return result, gr.update(choices=choices, value="All"), render(result, "All")


with gr.Blocks(title="Customers say") as demo:
    gr.Markdown("## Customers say\nAspect highlights extracted by RoBERTa-ABSA (R-Drop).")
    reviews_box = gr.Textbox(value=EXAMPLE_REVIEWS, lines=8, label="Amazon reviews (one per line)")
    analyze_btn = gr.Button("Analyze Reviews", variant="primary")
    chips = gr.Radio(choices=["All"], value="All", label="Select to learn more")
    output = gr.HTML(render(None, "All"))
    state = gr.State(None)

    analyze_btn.click(analyze, reviews_box, [state, chips, output])
    chips.change(render, [state, chips], output)

if __name__ == "__main__":
    demo.launch()
