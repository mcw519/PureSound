import sys
from common.plotter import Plotter

plotter = Plotter(verbose=True)

if len(sys.argv) < 2:
    sys.exit("File name not given. Please specify the file to plot")

file_name = sys.argv[1]
plotter.set_data_from_file(sys.argv[1])
plotter.plot()
