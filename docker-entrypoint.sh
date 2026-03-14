#!/bin/bash

source /opt/venv/bin/activate
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1

if [ -z "$1" ]; then
    set -- "bash"
fi

exec "$@"
