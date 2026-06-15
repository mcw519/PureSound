from matplotlib import pyplot as plt

'''
Companion script for verify_1D_fdtd_accuracy.py.
Creates a bar plot highlighting differences between each accuracy.
'''

plt.bar(8.97198486328125, label="Genauigkeit = 4")
plt.bar(8.896625137329101, label="Genauigkeit = 6")
plt.bar(8.693045711517334, label="Genauigkeit = 10")

plt.ylabel("Zeit [s]")
plt.xlabel("Iterationen")

plt.legend()
plt.show()