from scipy.io.wavfile import read, write
import numpy as np
import matplotlib.pyplot as plt
import sys

def wav_diff(filename1, filename2, output_file_name):
    fsl, left = read(filename1)
    _, right = read(filename2)

    left = np.array(left, dtype=np.float)
    right = np.array(right, dtype=np.float)

    diff = []

    for i in range(0, len(left)):
        diff.append(left[i] - right[i])

    diff = np.array(diff)

    write(output_file_name, fsl, diff.astype(np.float))

def visualize_multiple_waveforms(paths, dB=False):    
    signals = []
    times = []

    for path in paths:
        f_rate, x = read(path)
        signal = np.array(x, dtype=np.float)
        signals.append(signal)

        time = np.linspace(
            0, # start
            len(signal) / f_rate,
            num = len(signal)
        )

        times.append(time)

    fig = plt.figure()
    gs = fig.add_gridspec(len(paths), hspace=.01)
    axs = gs.subplots(sharex=True, sharey=True)

    for i in range(len(paths)):
        
        if dB:
            signal_to_plot = []
            for signal in signals[i]:
                if signal != 0:
                    signal_to_plot.append(20 * np.log10(np.abs(signal)))
                else:
                    signal_to_plot.append(0)
        else:
            signal_to_plot = signals[i]
        axs[i].plot(times[i], signal_to_plot, 'r-')
        axs[i].set_ylabel(paths[i], rotation=0, labelpad=20, fontsize=16)
        axs[i].get_yaxis().set_label_coords(-0.09,0.3)
        axs[i].tick_params(axis='both', which='major', labelsize=13)
        axs[i].tick_params(axis='both', which='minor', labelsize=10)

        axs[i].grid()

    plt.xlabel("Zeit [s]", fontsize=16)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit("Wave file names not given. Please specify two file name to compare and another file name to write the difference to.")

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_file = sys.argv[3]
    
    wav_diff(file1, file2, output_file)
    visualize_multiple_waveforms([file1, file2, output_file])
