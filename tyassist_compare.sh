#!/bin/bash

if [ "$1" ]; then
  TARGET="--target $1"
else
  echo "[warn] you didn't specify target in command, that will use the target specified in config file."
fi

mkdir -p logs
LOG_FILE="logs/tyassist_compare_$1_$(date "+%Y%m%d_%H%M%S").log"
echo "python3 $DENGINE_ROOT/tyassist/tyassist.py compare $TARGET -c config.yml 2>&1 | tee $LOG_FILE"
python3 $DENGINE_ROOT/tyassist/tyassist.py compare $TARGET -c config.yml 2>&1 | tee $LOG_FILE