# Use an official Python runtime as a parent image, suitable for TensorFlow
FROM tensorflow/tensorflow:latest

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and WeasyPrint
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info

# Install FastAPI and Uvicorn
RUN pip install fastapi uvicorn python-multipart aiofiles Pillow

# Copy the local directory contents into the container
COPY . /app

# Install `doctr` with TensorFlow support
RUN pip install python-doctr[tf]

# Expose the port FastAPI will run on
EXPOSE 8001

# Command to run the FastAPI server on container start
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "4"]