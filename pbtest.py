#!/bin/bash
for ((i = 0; i < 255; i++)); do
    ./run -p 0 1
    ./run -p 0 180
done
