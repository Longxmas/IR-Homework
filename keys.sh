#!/usr/bin/env bash

# the key used for debugging
test_key="sk-"

# one or multiple keys used for running experiments
declare -a keys=(
    "sk-"
)


export OPENAI_BASE_URL="https://api.openai-proxy.org/v1"
echo $OPENAI_API_KEY
