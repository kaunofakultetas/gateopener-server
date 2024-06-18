FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install flask

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["python3", "-u", "main.py"]
