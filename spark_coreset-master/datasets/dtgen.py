#!/usr/bin/env python
import random
import sys
__author__ = 'Anton & Ahmad'

x = 1
y = int(sys.argv[1])
if y > 0: x = y

d = 2
n = 300000*x
m = n / 1000
dist = 1000000
r = 3

f = open("dataset.txt",'w')
s = ""
for j in range(0, d-1):
	s += str(round(random.uniform(0,1000), r))+" "
s += str(round(random.uniform(0,1000), r))
f.write(s)
for i in range(0, n/m):
	s = ""
	for j in range(0, d-1):
		s += str(round(random.uniform(0,1000), r))+" "
	s += str(round(random.uniform(0,1000), r))
	f.write("\n"+s)

num = n-n/m
for i in range(0, num):
	s = ""
	for j in range(0, d-1):
		s += str(round(random.uniform(1000*dist,1100*dist), r))+" "
	s += str(round(random.uniform(1000*dist,1100*dist), r))
	f.write("\n"+s)

f.close()

