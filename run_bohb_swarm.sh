#!/usr/bin/env bash

NUM_WORKERS=6

RUN_ID=$1


set -m
set -e

python3 bohb_worker.py --run-id $* master &
sleep 5
echo 'Starting workers'

for ((k=0; k<NUM_WORKERS; k++))
do
  echo "starting worker $k"
  ./bohb_worker.py --run-id $RUN_ID worker &
done

trap "kill $(jobs -p)" EXIT

fg %1
