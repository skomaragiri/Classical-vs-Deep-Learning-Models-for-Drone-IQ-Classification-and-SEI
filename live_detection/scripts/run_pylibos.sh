#!/bin/bash
cd /home/polyseclab-g/Documents/sei_app

python pylibos_watch_dir.py \
    --watchdir    /home/polyseclab-g/Documents/automated_recordings \
    --output-path /home/polyseclab-g/Documents/annotated \
    --runtime     /home/polyseclab-g/Documents/sei_app/omnisig_model-2026-03-15-03-05.ds \
    --threshold   0.75 \
    --verbose
