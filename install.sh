# Install torch first as it's required for building xformers
# uv pip install torch torchvision torchaudio
uv pip install unsloth -U
# uv pip install transformers -U 
uv pip install -e ./
