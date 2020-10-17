from typing import List, Tuple

i = 0
curriculum = []

for line in open("rollins.txt", "r"):
    if (i%3 == 1):
        #TODO: handle objects
        print(line)
    elif (i%3 == 0):
        #TODO: handle language
        print(line)
    i += 1