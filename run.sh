#!/bin/sh

for i in {3..32} ; do
    python bocs/ohe/milp_time.py ${i} 0 3
done
