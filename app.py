from flask import Flask, render_template, request, send_file
from PIL import Image
import numpy as np
import torch
import os
from io import BytesIO
import cv2
from realesrgan import RealESRGAN
from gfpgan import GFPGANer

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Baixa modelo se necessário
if not os.path.exists("RealESRGAN_x4.pth"):
    os.system("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/RealESRGAN_x4.pth")

# Carrega modelos
sr_model = RealESRGAN(device, scale=4)
sr_model.load_weights("RealESRGAN_x4.pth")

gfpgan = GFPGANer(
    model_path=None,
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
    device=device
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        scale = int(request.form.get("scale", 4))
        restore_faces = "restore_faces" in request.form

        img = Image.open(file.stream).convert("RGB")
        img_np = np.array(img)

        if restore_faces:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            _, _, restored_bgr = gfpgan.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
            img_np = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_np)

        sr = RealESRGAN(device, scale=scale)
        sr.load_weights(f"RealESRGAN_x{scale}.pth")
        output_img = sr.predict(img)

        output_path = "static/output.png"
        output_img.save(output_path)

        return render_template("index.html", result=output_path)

    return render_template("index.html", result=None)

@app.route("/download")
def download():
    return send_file("static/output.png", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
