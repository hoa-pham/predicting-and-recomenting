
f = open("numpy_formatted.txt", "r")
lines = f.readlines()

f.close()

f = open("numpy_formatted.txt", "w")
for line in lines:
  if "?" not in line:
    f.write(line)
f.close()