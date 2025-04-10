# Install main packages
conda create -n wico-llava python=3.10 -y
conda activate wico-llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# Install extra packages
pip install -e ".[train]"
pip install flash-attn==2.5.3 --no-build-isolation  
