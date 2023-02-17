#!/bin/sh

for i in `seq 0 49`
do
  python bocs/annealings/qa.py base.id=${i} &
done
