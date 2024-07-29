from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import rembg
import time
import logging
import sys
from TripoSR.tsr.system import TSR
from TripoSR.tsr.utils import remove_background, resize_foreground, save_video
from TripoSR.tsr.bake_texture import bake_texture

# Your imports for TripoSR and other utilities here

app = Flask(__name__)
app.config['UPLOAD_FOLDER']='static/generated_images'
# Ensure the output directory exists
output_dir = app.config['UPLOAD_FOLDER']
os.makedirs(output_dir, exist_ok=True)
# Global variables and configurations
#output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)
sys.path.append('./TripoSR')
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
pretrained_model_name_or_path = "stabilityai/TripoSR"
chunk_size = 8192
mc_resolution = 256
no_remove_bg = False
foreground_ratio = 0.85
model_save_format = "obj"
bake_texture_flag = False
texture_resolution = 2048
render_flag = False

model = TSR.from_pretrained(
    pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)

# Timer class definition

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

timer = Timer()
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        num_samples = int(request.form['num_samples'])
        return redirect(url_for('generate_images', prompt=prompt, num_samples=num_samples))
    return render_template('index.html')

@app.route('/generate_images', methods=['GET', 'POST'])
def generate_images():
    prompt = request.args.get('prompt')
    num_samples = int(request.args.get('num_samples'))

    # Generate images
    images = []
    for _ in range(num_samples):
        with torch.autocast("cuda"):
            image = pipe(prompt).images[0]
            images.append(image)

    # Save images to disk
    img_paths = []
    for idx, img in enumerate(images):
        img_path = os.path.join(output_dir, f"image_{idx}.png")
        img.save(img_path)
        print(img_path)
        img_paths.append(img_path)

    if request.method == 'POST':
        selected_image = request.form['selected_image']
        model_format = request.form['model_format']
        return redirect(url_for('generate_3d_model', img_path=selected_image, model_format=model_format))

    return render_template('select_image.html', images=img_paths)

@app.route('/generate_3d_model', methods=['GET', 'POST'])
def generate_3d_model():
    img_path = request.args.get('img_path')
    model_format = request.args.get('model_format')

    image = Image.open(img_path)

    # Process image with TripoSR
    if no_remove_bg:
        img = np.array(image.convert("RGB"))
    else:
        img = remove_background(image, rembg.new_session())
        img = resize_foreground(img, foreground_ratio)
        img = np.array(img).astype(np.float32) / 255.0
        img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * 0.5
        img = Image.fromarray((img * 255.0).astype(np.uint8))

    # Run TripoSR model
    with torch.no_grad():
        scene_codes = model([img], device=device)

    meshes = model.extract_mesh(scene_codes, not bake_texture_flag, resolution=mc_resolution)
    output_path = os.path.join(output_dir, f"mesh.{model_format}")
    meshes[0].export(output_path)

    return render_template('download.html', mesh_path=output_path)

@app.route('/download/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
