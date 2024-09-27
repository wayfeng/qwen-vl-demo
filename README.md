# Qwen-VL-demo
Qwen2-VL demo with OpenVINO

Setup environment.

```bash
python3 -m venv .env
source .env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run the demo.

```bash
GRADIO_SERVER_NAME=0.0.0.0 python demo.py
```
