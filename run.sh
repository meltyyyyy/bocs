#!/bin/sh

for i in {16..32} ; do
    python bocs/ohe/milp_time.py ${i} 0 3
done
