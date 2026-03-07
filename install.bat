@echo off
echo Installing Unsloth and dependencies...
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install -r requirements.txt
echo Done!
