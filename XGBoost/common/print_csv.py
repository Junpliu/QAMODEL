#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, os, re

minThr = float(sys.argv[1])
maxThr = float(sys.argv[2])
dim = ","
for line in sys.stdin:
    try:
        q1, q2, label, predict = line.strip().split(dim)
        pre = float(predict)
    except:
        continue

    if minThr < pre < maxThr:
        print("\t".join([predict, label, q1, q2]))
