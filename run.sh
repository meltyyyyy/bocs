#!/bin/sh

for i in {3..15} ; do
    python3 exps/create_study.py milp --n_vars ${i} --n_runs 10
done
