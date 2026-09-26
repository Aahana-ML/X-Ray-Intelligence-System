from io import BytesIO

import numpy as np
import matplotlib.cm as cm

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.inference import (
    predict_image,
    class_names
)

from app.inference import model

from app.gradcam import (
    create_grad_model,
    make_gradcam
)


app = FastAPI(
    title="X-Ray Intelligence System",
    description="Multi-label chest X-ray abnormality prediction API",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


grad_model = create_grad_model(model)






@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    image, results = await predict_image(file)

    return {
        "filename": file.filename,
        "predictions": results
    }


@app.post("/explain")
async def explain(
    file: UploadFile = File(...),
    class_name: str = Form(...)
):

    if not file.content_type.startswith("image/"): 
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
    )

    if class_name not in class_names:
        raise HTTPException(
            status_code=400,
            detail=f'Unknown class. Choose one of: {class_names}'
        ) 

    image, results = await predict_image(file)

    class_index = class_names.index(class_name)

    heatmap = make_gradcam(
        grad_model,
        image,
        class_index
    )

    original = image[0].numpy()
    original = original / 255.0

    colored_heatmap = cm.jet(heatmap)[..., :3]

    overlay = (
        0.6 * original +
        0.4 * colored_heatmap
    )

    overlay = np.clip(
        overlay * 255,
        0,
        255
    ).astype(np.uint8)

    from PIL import Image

    overlay_image = Image.fromarray(overlay)

    buffer = BytesIO()

    overlay_image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="image/png"
    )


app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "frontend"),
    name="static"
)

@app.get("/")
def frontend():
    return FileResponse(BASE_DIR / "frontend" / "index.html")