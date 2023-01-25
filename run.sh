#!/bin/sh


array=(
        # 5 \
        # 6 \
        # 7 \
        # 8 \
        9
)

for i in ${array[@]}
do
  for j in `seq 0 0`
  do
    python bocs/annealings/sqa_qpu_milp.py ${i} 0 3 ${j}
  done
done
