#!/bin/bash
for i in {1..4}; do
  echo -e "\nROUND $i\n"
  for j in {1..3}; do
    python3 train_and_test.py &
  done
  wait
done >>results.txt 
