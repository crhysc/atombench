#!/bin/bash

cd models/atomgpt
conda create -y -n my_atomgpt python=3.10
conda run -n my_atomgpt python -m pip install --upgrade pip
conda run -n my_atomgpt pip install uv
conda run -n my_atomgpt uv pip install -e .
