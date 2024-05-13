FROM anibali/pytorch:2.0.1-cuda11.8-ubuntu22.04

# RUN sudo apt-get update
# RUN sudo apt-get upgrade -y
# RUN sudo apt-get install -y \
#         build-essential 

RUN sudo apt-get update \
 && sudo apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && sudo rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/.

RUN pip install -r requirements.txt

