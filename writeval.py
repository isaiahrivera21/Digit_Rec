
index = 12
psi = 222
kpa = 34
f = open("Generic.txt", "a")


for i in range(12):
    f.write("{} , {} , {} \n".format(index, psi, kpa))
f.close()

