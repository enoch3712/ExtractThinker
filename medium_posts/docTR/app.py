from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from PIL import Image
import io
import os

app = FastAPI(title="OCR Service using docTR")

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    image_data = await file.read()

    # Attempt to load the image directly from the BytesIO object
    doc = DocumentFile.from_images(image_data)

    model = ocr_predictor(pretrained=True)
    result = model(doc)

    extracted_texts = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join([word.value for word in line.words])
                extracted_texts.append(line_text)

    return JSONResponse(content={"ExtractedText": extracted_texts})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)