#! /usr/bin/bash

count=0
total_runs=10
for i in $(seq 1 $total_runs)
do
    python3 rigid_hom.py
    if [ $? -ne 0 ]; then
        (( count += 1 ))
    fi
done

echo "Number of failures / total runs : $count / $total_runs"
