#!/bin/sh

for i in {7..16}
do
  python bocs/ohe/milp_time.py ${i} 0 3
done
