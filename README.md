# LiquidAI
README – Liquid AI Audio Chat Demo

This repository provides a minimal, reproducible example demonstrating how to run an audio-based conversation using the Liquid AI LFM2-Audio-1.5B model.
The project includes Python scripts, a Colab notebook, and sample audio files to help you test the model quickly on any GPU-enabled setup.

1. Requirements

The demo requires an NVIDIA GPU with CUDA support.

If you do not have a local NVIDIA GPU, you can run the project using the included Google Colab notebook, which has a fully configured environment and works out of the box with Colab’s GPU runtime.

2. Environment Setup (Conda – Local GPU Execution)

Follow these steps to set up a clean and isolated environment on your local machine.

Step 1 — Install Conda

Download Miniconda or Anaconda:
https://docs.conda.io/en/latest/miniconda.html

Step 2 — Create a dedicated environment
conda create --name liquid python=3.12

Step 3 — Activate the environment
conda activate liquid

Step 4 — Install all required packages

All dependencies are listed in requirements.txt:

pip install -r requirements.txt

Optional — Enable CUDA debugging

If needed, add this at the top of the script:

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

3. Running the Demo (Local GPU)

The main script is located in the Liquid_AI directory (liquid_demo.py).
To execute the full audio conversation pipeline:

python liquid_demo.py


Make sure you update the audio file paths inside the script according to your local directory structure.

4. Running the Demo on Google Colab (GPU Runtime Enabled)

A ready-to-use Jupyter notebook is provided in the repository.
It already includes:

environment setup

required installations

model loading

full audio conversation workflow

How to use it:

Open the notebook in Google Colab

Go to Runtime → Change runtime type

Select GPU

Run all cells sequentially

This method requires no local installation and is ideal for quick testing or for users without a local CUDA-enabled device.

5. Input Audio Files

Three example audio recordings are included in the Liquid_Q&A directory.
They are already referenced in the script; you only need to adjust their absolute paths.

Replace them freely with your own .wav or .m4a files.

Example:

wav, sampling_rate = torchaudio.load(
    "C:/Users/.../Liquid_Q&A/Liquid_question.m4a"
)

6. Output and Evaluation

Each conversation turn produces two outputs:

1. Printed text

Generated tokens are streamed in real time to the terminal.

2. Audio response

The model generates audio that is decoded and saved as .wav files:

answer_Liquid.wav

answer_GPU.wav

answer_Cuda.wav

answer_solution.wav

You can evaluate the results by:

Listening to the generated .wav files

Reviewing the printed text response

Inspecting the flow of turns in the terminal

7. Repository Structure
Liquid_AI/          → Main Python script (liquid_demo.py)
Liquid_Q&A/         → Sample audio questions
Liquid_Colab/       → Google Colab notebook for GPU execution
Packages/           → Requirements list
README.md           → Documentation

8. Notes and Recommendations

A GPU with CUDA is required for reliable performance.

Use the Colab notebook if you do not have access to a local NVIDIA GPU.

Always verify that your audio file paths are correct before running the script.

The model is large; expect slow startup times on smaller GPUs.
