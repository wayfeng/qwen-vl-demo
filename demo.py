from pathlib import Path
from gradio_helper import make_demo
from ov_qwen2_vl import OVQwen2VLModel
from transformers import AutoProcessor, AutoTokenizer

model_id = "Qwen/Qwen2-VL-7B-Instruct"
model_dir = Path(model_id.split("/")[-1])
model = OVQwen2VLModel(model_dir=model_dir, device="GPU.1")

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(model_dir, min_pixels=min_pixels, max_pixels=max_pixels)
if processor.chat_template is None:
    tok = AutoTokenizer.from_pretrained(model_dir)
    processor.chat_template = tok.chat_template

demo = make_demo(model, processor)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(debug=True, share=True)