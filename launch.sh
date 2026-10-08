#!/bin/bash

python3 src/setup.py
python3 src/trainer.py
python3 src/calculate.py

echo "Готово"
