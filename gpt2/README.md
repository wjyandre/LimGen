# GPT2

## Installation

Install tensorflow 1.12 (with GPU support, if you have a GPU and want everything to run faster)
```
pip3 install tensorflow==1.12.0
```
or
```
pip3 install tensorflow-gpu==1.12.0
```

Install other python packages:
```
pip3 install -r requirements.txt
```

Create a folder named 'models', and download the model data:
```
python3 download_model.py 117M
```
## Unconditional sample generation

To generate unconditional samples from the small model:
```
python3 src/generate_unconditional_samples.py | tee /tmp/samples
```
There are various flags for controlling the samples:
```
python3 src/generate_unconditional_samples.py --top_k 40 --temperature 0.7 | tee /tmp/samples
```

To check flag descriptions, use:
```
python3 src/generate_unconditional_samples.py -- --help
```

## Conditional sample generation

Generate conditional text by:
```
python3 src/interactive_conditional_samples.py 40
```

To give the model custom prompts, you can use:
```
python3 src/interactive_conditional_samples.py --top_k 40
```
