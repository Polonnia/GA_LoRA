# Apply this add-on

From the extracted add-on directory:

```bash
cp -r gpt2_lora /path/to/GA_LoRA/
cp requirements_gpt2.txt GPT2_EXPERIMENTS.md /path/to/GA_LoRA/
cd /path/to/GA_LoRA
pip install -r requirements_gpt2.txt
bash gpt2_lora/run_gpt2.sh
```

No existing CLIP/GA-LoRA file is overwritten. The language experiment is isolated in a new Python package.
