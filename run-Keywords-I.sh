#!/bin/bash
/opt/microsoft/mlserver/9.3.0/runtime/python/bin/jupyter nbconvert --execute --to=html Keywords-I-Anomaly-Detection.ipynb
cp Keywords-I-Anomaly-Detection.html results/Keywords-I-Anomaly-Detection_$(date +%Y-%m-%d).html
