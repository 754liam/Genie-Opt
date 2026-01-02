FROM python:3.9-slim
# Base Linux OS with Python 3.9 installed (slim for speed)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# CPP compilers; graphics libraries - not in requirements.txt as these are system tools
WORKDIR /app
#Containerizes program inside an app directory to isolate it from system files.
COPY requirements.txt .

RUN pip install pip==23.3.1 setuptools==65.5.0 wheel==0.38.4


RUN pip install "numpy<1.24"

RUN pip install --no-build-isolation gym==0.21.0

# Smart layer caching - a seperation of the dependencies from the program; useful so Docker doesn't have to copy requirements.txt again when code is adjusted.
RUN pip install --no-cache-dir -r requirements.txt
# Runs pip install with no storage system 
COPY . .
# Copies everything from this projects root directory over to current root inside the container (in this case it'll be /app)
CMD ["python", "data_collection.py"]
# Run command 

# Note: the docker build takes a while when first run as PyTorch is massive. Once run once however, it'll be extremely fast for all future executions.