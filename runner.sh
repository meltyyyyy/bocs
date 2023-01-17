#!/bin/sh


array=(
        5 \
        6 \
        7 \
        8 \
        9 \
)

for i in ${array[@]}
do
  for j in `seq 0 99`
  do
    python bocs/annealings/sqa_miqp.py ${i} 0 3 ${j}
  done
done
