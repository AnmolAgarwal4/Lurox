#!/bin/bash
echo "Building Lurox C engine..."
gcc -shared -fPIC -o Core/lurox_core.so Core/index.c

echo "Building embeddings..."
python scripts/build_embeddings.py

echo "Running alpha sweep..."
python scripts/alpha_sweep.py

echo "Running full evaluation..."
python scripts/full_evaluation.py

echo "Generating charts..."
python benchmarks/generate_charts.py

echo "Done. Results in benchmarks/"