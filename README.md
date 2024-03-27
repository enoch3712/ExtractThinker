# Open-DocLLM

## Introduction
This project aims to tackle the challenges of data extraction and processing using OCR and LLM. It is inspired by JP Morgan's DocLLM but is fully open-source and offers a larger context window size. The project is divided into two parts: the OCR and LLM layer.

![image](https://github.com/enoch3712/Open-DocLLM/assets/9283394/2612cc9e-fc66-401e-912d-3acaef42d9cc)

## OCR Layer
The OCR layer is responsible for reading all the content from a document. It involves the following steps:

1. **Convert pages to images**: Any type of file is converted into an image so that all the content in the document can be read.

2. **Preprocess image for OCR**: The image is adjusted to improve its quality and readability.

3. **Tesseract OCR**: The Tesseract OCR, the most popular open-source OCR in the world, is used to read the content from the images.

## LLM Layer
The LLM layer is responsible for extracting specific content from the document in a structured way. It involves defining an extraction contract and extracting the JSON data.

## Running Locally
You can run the models on-premises using LLM studio or Ollama. This project uses LlamaIndex and Ollama.

## Testing
The repo includes a FastAPI app with one endpoint for testing. Make sure to point to the proper Tesseract executable and change the key in the config.py file.

## Advanced Cases: 1 Million token context
The project also explores advanced cases like a 1 million token context using LLM Lingua and Mistral Yarn 128k context window.

## Conclusion
The integration of OCR and LLM technologies in this project marks a pivotal advancement in analyzing unstructured data. The combination of open-source projects like Tesseract and Mistral makes a perfect implementation that could be used in an on-premise use case.

## References & Documents 
1. [DOCLLM: A LAYOUT-AWARE GENERATIVE LANGUAGE MODEL FOR MULTIMODAL DOCUMENT UNDERSTANDING](https://arxiv.org/pdf/2401.00908.pdf)
2. [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/pdf/2309.00071.pdf)