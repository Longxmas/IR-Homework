# Active Spec RAG

## Overview

Active Spec RAG is a generic retrieval-augmented generation method that actively decides when and what to retrieve using a prediction of the upcoming sentence to anticipate future content and utilize it as the query to retrieve relevant documents if it contains low-confidence tokens. During generation, it utilizes a smaller specialist LM to generate draft texts, which are then fed to a larger generalist LM for verification and selection of the best draft. 

<p align="center">
  <img align="middle" src="res/design.png" height="350" alt="AS-RAG"/>
</p>

## Install environment with Conda
Create a conda env and follow `setup.sh` to install dependencies.

## Quick start

### Download Wikipedia dump
Download the Wikipedia dump from [the DPR repository](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command:
```shell
mkdir data/dpr
wget -O data/dpr/psgs_w100.tsv.gz https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
pushd data/dpr
gzip -d psgs_w100.tsv.gz
popd
```

### Build Wikipedia index
We use Elasticsearch to index the Wikipedia dump.
```shell
wget -O elasticsearch-7.17.9.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.17.9-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-7.17.9.tar.gz
pushd elasticsearch-7.17.9
nohup bin/elasticsearch &  # run Elasticsearch in background
popd
python prep.py --task build_elasticsearch --inp data/dpr/psgs_w100.tsv wikipedia_dpr  # build index
```

### Setup Bing search
This is only required for experiments on the WikiASP dataset.
1. Create a bing search API key following instructions on [https://www.microsoft.com/en-us/bing/apis/bing-web-search-api](https://www.microsoft.com/en-us/bing/apis/bing-web-search-api).
2. Run a local bing search server with caching functionality to save credits: `export BING_SEARCH_KEY=$YOUR_KEY; python bing_search_cache_server.py &> bing_log.out &`.

### Setup OpenAI keys
Put OpenAI keys in the `keys.sh` file.
Multiple keys can be used to accelerate experiments.
Please avoid uploading your keys to Github by accident!

### Run Active Spec RAG
Use the following command to run AS-RAG with `gpt-4o-mini`. 
```shell
./openai.sh 2wikihop configs/2wikihop_config.json  # 2WikiMultihopQA dataset
```
Be careful, experiments are relatively expensive because AS-RAG calls OpenAI API multiple times for a single example. You can decrease `max_num_examples` to run small-scale experiments to save credits.
Set `debug=true` to active the debugging mode which walks you through the iterative retrieval and generation process one example at a time.
