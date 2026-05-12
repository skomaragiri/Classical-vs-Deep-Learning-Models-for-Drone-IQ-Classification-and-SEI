#!/bin/bash
cd /home/polyseclab-g/Documents/sei_app

python gnuradio_watcher.py \
    --watchdir     /home/polyseclab-g/Documents/annotated \
    --output-path  /home/polyseclab-g/Documents/sei_watcher_output \
    --model        ./sei_model_n11_ul \
    --label-filter n11_pro_UL \
    --target-rate  10e6 \
    --profiles     ./emitter_profiles.json \
    --fuse-capture \
    --enable-zmq --port 9200 \
    --verbose
