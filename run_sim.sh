#!/bin/bash
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LD_LIBRARY_PATH=/home/janvi/CoppeliaSim_Edu_V4_9_0_rev6_Ubuntu24_04:$LD_LIBRARY_PATH \
python sim/sim.py
