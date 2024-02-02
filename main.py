from http.client import HTTPException
import os
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import validators
from uvicorn import run
from CustomException import CustomException
from PyPDF2 import PdfReader
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
from pathlib import Path
import tempfile
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
import PyPDF2
from easyocr import Reader
import pytesseract
from pytesseract import image_to_string
import pypdfium2 as pdf2img
import pypdfium2 as pdfium
from PIL import Image
import aiofiles
import os
import asyncio
import time
import concurrent.futures
import imghdr
import urllib.request
import io
from pyzbar.pyzbar import decode
import pandas as pd
import requests
import json
from typing import Dict
from Payload import Message, Payload
from config import API_KEY

# local path to tesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
#os.environ.get('TESSERACT_PATH', 'tesseract')

systemMessage = "You are a server API that receives document information and returns specific document fields as a JSON object."
jsonContentStarter = "```json"

app = FastAPI()

@app.get("/error")
async def error_route():
    try:
        # Your code that may raise an exception
        raise CustomException("Something went wrong")
    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))
    
async def delete_previous_files():
    tempFilPath = Path(tempfile.gettempdir())
    current_time = time.time()
    for file in tempFilPath.iterdir():
        try:
            if os.path.isfile(file) and (current_time - os.path.getctime(file)) // 60 > 1:
                os.remove(file)
        except:
            pass

@app.post("/extract_text")
async def extract_text(file: UploadFile = File(...)):

    # Create a temporary file and save the uploaded file to it
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    shutil.copyfileobj(file.file, temp_file)
    file_path = temp_file.name
    
    # Convert PDF to images
    images = convert_pdf_to_images(file_path)
    
    # Extract text using different methods
    result = {
        "data": extract_text_with_pytesseract(images),
    }

    # Close and remove the temporary file
    temp_file.close()

    return JSONResponse(content=result)

@app.post("/extract_text_from_url")
async def extract_text_from_url(url: str):
    # Validate the URL
    if not validators.url(url):
        raise HTTPException(status_code=400, detail="Invalid URL")

    # Download the file
    temp_file_path, _ = urllib.request.urlretrieve(url)

    # Convert PDF to images
    images = convert_pdf_to_images(temp_file_path)
    
    # Extract text using different methods
    result = {
        "data": extract_text_with_pytesseract(images),
    }

    return JSONResponse(content=result)

@app.post("/extract")
async def extract_text(file: UploadFile = File(...), extraction_contract: str = Form(...)):

    # Create a temporary file and save the uploaded file to it
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    shutil.copyfileobj(file.file, temp_file)
    file_path = temp_file.name
    
    # Convert PDF to images
    images = convert_pdf_to_images(file_path)
    
    # Extract text using different methods
    extracted_text = extract_text_with_pytesseract(images)

    # Join the extracted text into a single string
    extracted_text = "\n new page --- \n".join(extracted_text)

    # add system message to the extracted text
    extracted_text = systemMessage + "\n####Content\n\n" + extracted_text

    # add contract to the extracted text
    extracted_text = extracted_text + "\n####Structure of the JSON output file\n\n" + extraction_contract

    # add response section
    extracted_text = extracted_text + "\n#### JSON Response\n\n" + jsonContentStarter

    # Send the extracted text and extraction contract to the Mistral API
    content = send_request_to_mistral(extracted_text)

    # 

    # Close and remove the temporary file
    temp_file.close()

    return content

def send_request_to_mistral(content: str) -> str:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }

    message = Message(role="user", content=content)
    payload = Payload(
        model="mistral-tiny",
        messages=[message.dict()],
        temperature=0.7,
        top_p=1,
        max_tokens=2000,
        stream=False,
        safe_prompt=False
    )

    response = requests.post(url, headers=headers, data=payload.json().encode('utf-8'))
    response_json = response.json()

    # Extract the content from the response and convert it to a string
    content = str(response_json.get('choices', [{}])[0].get('message', {}).get('content', ''))

    # Extract the JSON substring from the content
    json_content = extractJsonSubstring(jsonContentStarter, content)

    # deserialize the JSON string
    json_content = json.loads(json_content)

    return json_content

def extractJsonSubstring(str1, str2):
    # Concatenate the two strings
    combined_str = str1 + str2
    
    # Define the start and end markers
    start_marker = "```json"
    end_marker = "```"
    
    # Find the start and end positions of the substring
    start_pos = combined_str.find(start_marker)
    ### jump start_pos to the end of the start_marker
    start_pos = start_pos + len(start_marker)
    end_pos = combined_str.find(end_marker, start_pos)
    
    # Extract the substring
    json_substring = combined_str[start_pos:end_pos]

    return json_substring

def convert_pdf_to_images(file_path, scale=300/72):
    # Check if the file is already an image
    if imghdr.what(file_path) is not None:
        # If it is, return it as is
        with open(file_path, 'rb') as f:
            return [{0: f.read()}]

    # If it's not an image, proceed with the conversion
    pdf_file = pdfium.PdfDocument(file_path)
    
    page_indices = [i for i in range(len(pdf_file))]
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in page_indices:
            future = executor.submit(render_page, pdf_file, i, scale)
            futures.append(future)
        
        final_images = []
        for future in concurrent.futures.as_completed(futures):
            final_images.append(future.result())
    
    return final_images

def render_page(pdf_file, page_index, scale):
    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=[page_index],
        scale=scale,
    )
    image_list = list(renderer)
    image = image_list[0]
    image_byte_array = BytesIO()
    image.save(image_byte_array, format='jpeg', optimize=True)
    image_byte_array = image_byte_array.getvalue()
    return {page_index: image_byte_array}

def extract_text_with_pytesseract(list_dict_final_images):
    
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for index, image_bytes in enumerate(image_list):
            future = executor.submit(process_image, index, image_bytes)
            futures.append(future)
        
        for future in concurrent.futures.as_completed(futures):
            try:
                raw_text = future.result()
                image_content.append(raw_text)
            except Exception as e:
                raise Exception(f"Error processing image: {e}")
    
    return image_content

def process_image(index, image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        return raw_text
    except Exception as e:
        raise Exception(f"Error processing image {index}: {e}")

def extract_text_with_easyocr(images):
    reader = Reader(["en"])
    text = ''
    for img in images:
        results = reader.readtext(img)
        for result in results:
            text += result[1] + ' '
    return text

def extract_text_with_langchain_image(images):
    text = ''
    for img in images:
        loader = UnstructuredImageLoader(img)
        text += loader.load()
    return text

def extract_text_with_langchain_pdf(pdf_file):
    loader = UnstructuredFileLoader(pdf_file)
    return loader.text

def extract_text_with_pyPDF(PDF_File):
    with open(PDF_File, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(pdf_reader.getNumPages()):
            text += pdf_reader.getPage(page_num).extractText()
    return text

# Define a coroutine that runs the extract_text() function every 5 minutes
async def run_periodically():
    while True:
        await delete_previous_files()
        await asyncio.sleep(30)  # Wait for 5 minutes

# Start the event loop and schedule the run_periodically() coroutine to run
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(run_periodically())
    loop.run_forever()