# Use an official Python 3.12 image from Docker Hub
FROM python:3.12-slim-bookworm

# Set the working directory
WORKDIR /app

# Copy your application code
COPY . /app

# Explicitly copy preprocessor
COPY preprocessor_obj /app/preprocessor_obj

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["python3", "app.py"]