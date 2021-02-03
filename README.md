# Tacl-Limerick-Generation
*LimGen* is an Automatic Limerick Generation system that uses Adaptive Multi-Template Constraints (AMTC), Multi-Templated Beam Search (MTBS), Storyline Algorithm, pre-trained small (345M) [GPT-2](https://openai.com/blog/better-language-models/) and various NLP tools. The paper *''There once was a really bad poet, it was automated but you didn’t know it''* that introduced *LimGen* is included in [TACL](https://transacl.org/index.php/tacl) 2021. This repository also includes modified previous state-of-the-art sonnet generation system ([DeepSpeare](https://www.aclweb.org/anthology/P18-1181/)). Various beam search baselines are also included.

# Requirements
- Recommend to create 2 virtual environments
- For LimGen with GPT2:
	- `conda create -n LimGen python=3.6 tensorflow==1.12.0`
	- `source activate LimGen`
	- For GPU usage in CUDA-enabled machines, `pip3 install tensorflow-gpu==1.12.0`
	- `conda install nomkl`
	- `pip3 install nltk`; `pip3 install gensim`; `pip3 install spacy`; `pip3 install fire`
	- `python3 -m spacy download en_core_web_lg`
	- `pip3 install regex==2017.4.5`; `pip3 install requests==2.21.0`; `pip3 install tqdm==4.31.1`
	- `conda deactivate`

- For DeepSpeare Baseline:
	- `conda create -n DS python=2.7 tensorflow==0.12.0`
	- `source activate DS`
	- `pip install nltk==3.0.0; import nltk ;nltk.download("cmudict"); nltk.download("stopwords")`
	- `pip install sklearn==0.20.4`; `pip install gensim`
	- `conda deactivate`

# Datasets / Pretrained Models
- To download pre-trained GPT2 (345M), `cd gpt2`. Create a folder named 'models', and download the model data: `python3 download_model.py 345M`

- To unzip pre-trained DeepSpeare Model, `cd deepspeare; tar -xvzf trained_model.tgz`.

- to unzip necessary data for LimGen and DeepSpeare.
	- `cd deepspeare; tar -xvzf saved_objects.tgz`
	- `cd LimGen; tar -xvzf saved_objects.tgz`

# Proposed Model: LimGen with Multi-Templated Beam Search (MTBS)
- LimGen system is located in `LimGen/`.
- Activate environment, `source activate LimGen`
- To produce limericks from a specific prompt_word in a specific directory, `python3 run_Limericks.py -p [prompt_word]  -dir [direcotory_name]`.
- Produce limericks using beam search baselines (not MTBS).
- `conda deactivate`

# Baseline Model: DeepSpeare for Limericks
- Modified DeepSpeare is located in `deepspeare/`
- Activate python2 environment, `source activate DS`
- Generate the body (last 4 lines) of NUM_SAMPLES number of limericks, results located in `deepspeare/deepspeare_results.pickle`:
```
python deepspeare/sonnet_gen.py -n NUM_SAMPLES
```
- Write first lines for these limericks using LimGen system, `conda deactivate; source activate LimGen; python run_DeepSpeare.py`, txt file of limericks located in `deepspeare_limericks/deepspeare_limericks.txt`.

# For Mass Producing in LimGen:
- Mass Producing limericks using LimGen in SLURM, refer to syntax of `submit_story_male.sh`.

