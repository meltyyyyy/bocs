#!/bin/sh


array=(
        8 \
        12 \
        16 \
        20 \
        24
)

for i in ${array[@]}
do
  python bocs/annealings/sa.py ${i} 0 3
done

for i in ${array[@]}
do
  python bocs/annealings/sqa.py ${i} 0 3
done
