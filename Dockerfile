# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in pyproject.toml
RUN pip install --upgrade pip \
	&& pip install setuptools wheel \
	&& pip install .[all]  # Assuming setup.py exists and handles dependencies correctly

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME EgoAllo

# Run app.py when the container launches
CMD ["python", "app.py"]