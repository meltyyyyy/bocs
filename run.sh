#!/bin/sh

for i in {1..32} ; do
    python bocs/ohe/milp_range.py 3 0 ${i}
done
