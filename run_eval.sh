export OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
export OPENAI_API_KEY="sk-"
python prep.py --task eval --inp /data/longlingkun/workspace/RAG/output/2wikihop/gpt-4o-mini/200_spec.jsonl --dataset 2wikihop --model gpt-4o-mini --out result.txt