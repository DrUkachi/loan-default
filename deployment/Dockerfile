# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Set W&B API key as an environment variable
# ENV WANDB_API_KEY your_wandb_api_key_here

# Run both FastAPI and Streamlit when the container launches
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 80 & streamlit run loan_prediction_ui.py --server.port 8501"]

