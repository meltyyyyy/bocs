#!/bin/sh

for i in {2..16} ; do
    python bocs/be/be_milp_range.py 3 0 $((2 ** i - 1))
done
