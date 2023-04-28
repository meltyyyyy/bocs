#!/bin/sh

for i in `seq 0 99`
do
  python bocs/annealings/miqp/sqa.py base.id=${i} base.exp="miqp" base.n_vars=4 &
done
