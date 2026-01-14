#!/bin/bash


python run_pure_python_ha.py \
    --input-data-path data_all/ATVA/ball \
    --manager-model gemini/gemini-flash-lite-latest \
    --max-iterations 2 \
    --optimization-iters 30

