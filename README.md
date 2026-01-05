# Genie-Opt

A reproduction of Google DeepMind's 'Genie' World Model, implementing its Spatiotemporal (ST) Transformer architecture to achieve linear O(T) computational complexity on consumer hardware.

## What it's all about

This project is an implementation of a primitive generative world model inspired from 'Genie'. As of writing, the data collection procedure and quantizer have been created, and the encoder/decoder are soon to come. To give a high level overview on how this works, 5000 frames of randomized action events are recorded through an observation object. This is all compiled into a 4D Tensor representing N frames of 64 x 64 x 3 shape. This is then passed into the encoder (not built yet), and then passed into the quantizer as a 16 x 16 x 64 4D tensor representing compressed frames with continuous description for each 'pixel'. A euclidean distance procedure is then performed which finds the vector of closest relation in the codebook to the encoders series of given continuous vectors for each frame. A one hot vector is then generated and matrix multiplication is performed to isolate the discrete vector suitable for describing the vector given from the codebook. A loss function is invoked which shifts the codebooks dictionary matrix towards the encoders output by a factor of 0.25 and vice versa by 1.0. This system is meant to keep the codebook/encoder relatively in line without too much restriction of information representation so a useful discrete vector can be formed from continuous mappings given from the encoder.



## How to Run

### Option 1: Using Docker (Recommended)

The easiest way to run this project is using Docker, which handles all dependencies automatically.

1. **Build the Docker image:**
   ```bash
   docker build -t genie-opt .
   ```
   Note: The first build may take a while as PyTorch is large (~2GB). Subsequent builds will be much faster due to layer caching.

2. **Run the data collection script:**
   ```bash
   docker run --rm genie-opt
   ```
   This will collect 5000 frames from the CoinRun environment and save them to `coinrun_data/training_data.npy`.

