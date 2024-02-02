# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN apt-get update
RUN apt-get update -y && apt-get install -y --no-install-recommends libzbar0
RUN apt-get install -y tesseract-ocr
RUN pip install -r ./requirements.txt

# Set the Tesseract path environment variable
ENV TESSERACT_PATH=/usr/bin/tesseract

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
