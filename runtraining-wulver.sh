#On WULVER:
# 1. Create a folder for your environment
python3 -m venv ~/torch_env

# 2. Activate it (you'll see the name in your prompt)
source ~/torch_env/bin/activate

# 3. Re-install torch inside this active environment
pip install torch

# 4. Run your script
python3 train_gpu.py
