Remember to clone the submodule before running the code:
```
git submodule update --init --recursive
```

## generate arc agi 2 visual data
```
python generateARCAGI2Visual.py
```
## start vllm server
```
vllm serve Qwen3.5-2B-Base --host 127.0.0.1 --port 8000 --served-model-name Qwen3.5-2B-Base --api-key 123
```
## inference with VLM
```
python vllmARCAGI.py --base-url http://127.0.0.1:8000/v1 --model Qwen3.5-2B-Base --api-key 123 --workers 120 --output-root data/baseTest
```
## evaluation
```
python rangeVote.py 1 120 1 --vote-root data/baseTestVote --existing-output-root data/baseTest
```
## print summary
```
python rangeVote.py 1 120 1 --vote-root data/baseTestVote --existing-output-root data/baseTest --summarize
```