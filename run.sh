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
  for j in `seq 0 10`
  do
    python bocs/annealings/qa.py base.id=${j}
  done
done
