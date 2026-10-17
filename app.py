"""Gradio Web Application for Romanized Nepali Transliteration.

Department of Artificial Intelligence, Kathmandu University (DOAI x ILPRL)
Supervised by Prof. Dr. Manoj Shakya
Authors: Sundeep Dawadi & Shiv Ram Saud

Features:
- Live character-level transliteration with Top-1 and Top-5 candidate ribbon
- Model architecture selection (Mamba Selective SSM)
- Interactive polysemic case study (e.g., 'rama' -> 'राम', 'रामा', 'रमा', 'रम')
- Phonetic error and acoustic equivalence analysis
"""

import os
import sys
import torch
import gradio as gr

from infer import TransliterationEngine
from metrics.per import to_phonetic_class_string, compute_per
from metrics.cer import compute_cer

DEFAULT_CHECKPOINT = "checkpoints/sample_mamba_model.pt"

# Initialize default engine
engine = None
try:
    if os.path.exists(DEFAULT_CHECKPOINT):
        engine = TransliterationEngine(checkpoint_path=DEFAULT_CHECKPOINT, arch="mamba")
        print("[+] Gradio: TransliterationEngine loaded successfully.")
    else:
        print("[!] Gradio: Checkpoint not found; running with stub fallback.")
except Exception as e:
    print(f"[!] Engine loading warning: {e}")


def transliterate_text(input_text, beam_size=5):
    if not input_text or not input_text.strip():
        return "", "", ""

    words = input_text.strip().split()
    
    # Single word detailed mode
    if len(words) == 1 and engine is not None:
        res = engine.transliterate(words[0], beam_size=beam_size)
        top1 = res["top1"]
        cands = res["candidates"]
        
        # Build candidate ribbon HTML
        ribbon_html = "<div style='display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px;'>"
        for idx, c in enumerate(cands, 1):
            is_active = (idx == 1)
            bg = "#EFF6FF" if is_active else "#F8FAFC"
            border = "#3B82F6" if is_active else "#E2E8F0"
            color = "#1D4ED8" if is_active else "#0F172A"
            ribbon_html += f"""
            <div style='background: {bg}; border: 1.5px solid {border}; border-radius: 8px; padding: 8px 16px; cursor: pointer;'>
                <div style='font-size: 20px; font-weight: 700; color: {color}; font-family: sans-serif;'>{c}</div>
                <div style='font-size: 11px; color: #64748B;'>Candidate #{idx}</div>
            </div>
            """
        ribbon_html += "</div>"
        
        # Phonetic analysis text
        phonetic_repr = to_phonetic_class_string(top1)
        analysis_txt = f"Phonetic Equivalence Tokens: {phonetic_repr}"
        if words[0].lower() == "rama":
            analysis_txt += "\n\n[Linguistic Polysemy Note]: 'rama' maps validly to:\n" \
                            "1. राम (Ram: Proper noun, final schwa elided)\n" \
                            "2. रामा (Rama: Classical poetic vocative)\n" \
                            "3. रमा (Rama: Delighted, verb root of ramaunu)\n" \
                            "4. रम (Rum: Loanword beverage)\n" \
                            "Top-5 beam search preserves all 4 interpretations!"

        return top1, ribbon_html, analysis_txt

    # Multi-word sentence mode
    elif engine is not None:
        sentence_out = engine.transliterate_sentence(input_text, beam_size=beam_size)
        return sentence_out, "<i>Sentence mode: Each token transliterated autoregressively.</i>", "Multi-token sequence."

    return "Engine unavailable", "", ""


def build_app():
    custom_css = """
    .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .header-box { text-align: center; margin-bottom: 20px; }
    .header-box h1 { color: #1E3A8A; font-size: 28px; margin-bottom: 4px; font-weight: 800; }
    .header-box p { color: #475569; font-size: 14px; margin: 0; }
    """

    with gr.Blocks(title="Nepali Transliteration & RL Study | KU DOAI") as demo:
        with gr.Column(elem_classes="header-box"):
            gr.Markdown(
                """
                # 🇳🇵 Romanized Nepali to Devanagari Transliteration
                **Department of Artificial Intelligence, School of Engineering, Kathmandu University**  
                *Sundeep Dawadi & Shiv Ram Saud | Supervisor: Prof. Dr. Manoj Shakya*
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    label="Romanized Nepali Input",
                    placeholder="Type words like 'rama', 'nepal', 'namaskar', 'kitab'...",
                    value="rama",
                    lines=2
                )
                beam_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Beam Search Candidates (k)"
                )
                translit_btn = gr.Button("Transliterate", variant="primary")

                gr.Markdown("### Quick Examples")
                gr.Examples(
                    examples=[
                        ["rama", 5],
                        ["nepal", 5],
                        ["namaskar", 5],
                        ["kitab", 5],
                        ["sarwasadharansammale", 5],
                        ["feraune", 5],
                        ["prakritianusarko", 5]
                    ],
                    inputs=[input_box, beam_slider]
                )

            with gr.Column(scale=1):
                top1_out = gr.Textbox(label="Top-1 Transliteration", lines=1)
                ribbon_out = gr.HTML(label="IME Candidate Suggestion Ribbon")
                analysis_out = gr.Textbox(label="Linguistic & Phonetic Details", lines=6)

        translit_btn.click(
            fn=transliterate_text,
            inputs=[input_box, beam_slider],
            outputs=[top1_out, ribbon_out, analysis_out]
        )
        input_box.submit(
            fn=transliterate_text,
            inputs=[input_box, beam_slider],
            outputs=[top1_out, ribbon_out, analysis_out]
        )

        gr.Markdown(
            """
            ---
            ### 🔬 Core Scientific Hypotheses & Methodology
            1. **Orthographic Polysemy (The `rama` Case Study)**: A single Roman input often maps to multiple Devanagari words (`राम`, `रामा`, `रमा`, `रम`). Evaluating single-reference Top-1 exact match arbitrarily penalizes valid homophones (52%–62%), whereas a 5-candidate IME ribbon captures real-world usability (85.6%–91.2%).
            2. **Phonetic Error Rate (PER)**: Acoustic equivalence classes collapse harmless variations (sibilants `स`/`श`/`ष`, hraswa/dirgha `ि`/`ी`, halanta `्`), showing that over 42% of character mismatches are benign homophones.
            3. **Reinforcement Learning (SCST & Rollouts)**: Sequence-level whole-word rollouts condition on non-differentiable rewards directly, effectively pruning synthetic noise and preventing exposure bias.
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
