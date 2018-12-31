#!/usr/bin/env bash

NUM_WORKERS=10

set -m

python3 bohb_worker.py --run-id tst1 master &
sleep 5
echo 'Starting workers'

for ((k=0; k<NUM_WORKERS; k++))
do
  echo "starting worker $k"
  ./bohb_worker.py --run-id tst1 worker &
done

trap "kill $(jobs -p)" EXIT

fg %1
