import os

# fo = open("projects/custom.txt","r")

# line = fo.readline()
# print(line)
l = []
# fo.close
with open("projects/custom.txt","r") as fo:
    line = fo.readline()
    l.append(str(line)[:len(line)-1])
    print(line)
    while line:
        l.append(str(line)[:len(line)-1])
        print(line)
        line = fo.readline()
print(l)