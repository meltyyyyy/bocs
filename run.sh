#!/bin/sh

for i in `seq 50 99`
do
  python bocs/ohe/milp/sblr_sa.py base.id=${i} base.exp="milp" base.n_vars=8 &
done
