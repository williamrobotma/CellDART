#!/usr/bin/env bash

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --inplace --execute CellDART_example_dlpfc_markers.ipynb
./eval.py
