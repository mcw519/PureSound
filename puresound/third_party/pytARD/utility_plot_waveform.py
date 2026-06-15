from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) < 2:
    sys.exit("File name not given. Please specify the file to plot")

file_name = sys.argv[1]
samplerate, y = wavfile.read(file_name)

plt.rc('font', size=18) #controls default text size
plt.rc('axes', titlesize=18) #fontsize of the title
plt.rc('axes', labelsize=18) #fontsize of the x and y labels
plt.rc('xtick', labelsize=18) #fontsize of the x tick labels
plt.rc('ytick', labelsize=18) #fontsize of the y tick labels
plt.rc('legend', fontsize=18) #fontsize of the legend
plt.plot(y, '-')
plt.grid()
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()