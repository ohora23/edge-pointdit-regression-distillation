#!/usr/bin/env python
"""Parse an rknn build stdout log and summarize NPU vs CPU op placement."""
import collections
import re
import sys

pat = re.compile(r"\]\s+\d+\s+(\w+)\s+(\w+)\s+(NPU|CPU)\b")
lines = open(sys.argv[1], errors="ignore").read().splitlines()
by = collections.Counter()
cpu = collections.Counter()
for ln in lines:
    m = pat.search(ln)
    if m:
        by[m.group(3)] += 1
        if m.group(3) == "CPU":
            cpu[m.group(1)] += 1
io = cpu.get("InputOperator", 0) + cpu.get("OutputOperator", 0)
real_cpu = by["CPU"] - io
print(f"NPU={by['NPU']} CPU={by['CPU']} (I/O boundary={io}, real-compute fallback={real_cpu})")
print("CPU ops:", dict(cpu))
