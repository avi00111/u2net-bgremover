import os
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import onnxruntime as ort

app = FastAPI(title="U2Net Background Remover")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

U2NET_MODEL_PATH = MODEL_DIR / "u2net.onnx"
SESSION = None

@app.on_event("startup")
def load_model():
    global SESSION
    if U2NET_MODEL_PATH.exists():
        SESSION = ort.InferenceSession(str(U2NET_MODEL_PATH))
        print(f"✅ Loaded U2Net model: {U2NET_MODEL_PATH.name}")
    else:
        print(f"⚠ U2Net model not found at {U2NET_MODEL_PATH}")

def refine_mask(mask: Image.Image, blur_radius: int = 3, threshold: int = 20) -> Image.Image:
    """Refine mask with blur + threshold + feathering"""
    # Smooth edges
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    # Convert to numpy for thresholding
    mask_np = np.array(mask).astype(np.uint8)

    # Threshold → remove weak noise
    mask_np = np.where(mask_np > threshold, mask_np, 0)

    # Back to PIL
    refined = Image.fromarray(mask_np.astype(np.uint8))

    # Feather edges (extra smoothness)
    refined = refined.filter(ImageFilter.GaussianBlur(1))

    return refined

def remove_bg_u2net(img_bytes: bytes, session: ort.InferenceSession):
    """Remove background using U2Net with refined mask"""
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    
    # Preprocess for U2Net
    img_resized = image.resize((320, 320))
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
    img_np = np.expand_dims(img_np, 0)        # NCHW

    # Run inference
    input_name = session.get_inputs()[0].name
    mask = session.run(None, {input_name: img_np})[0][0, 0]

    # Normalize mask
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size)

    # Refine mask
    mask = refine_mask(mask)

    # Apply alpha channel
    image = image.convert("RGBA")
    image.putalpha(mask)
    return image

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <body style="font-family: Arial; max-width: 600px; margin: auto;">
        <h2>U2Net Background Remover (High Quality)</h2>
        <form action="/remove-bg" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br><br>
            <button type="submit">Remove Background</button>
        </form>
    </body>
    </html>
    """

@app.post("/remove-bg")
async def api_remove_bg(file: UploadFile):
    if SESSION is None:
        return {"error": "U2Net model not loaded"}
    try:
        img_bytes = await file.read()
        result_img = remove_bg_u2net(img_bytes, SESSION)

        buf = BytesIO()
        result_img.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
