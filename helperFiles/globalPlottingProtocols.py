# General
import os

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt


class globalPlottingProtocols:

    def __init__(self):
        # Setup matplotlib
        self.saveDataFolder = None
        plt.rcdefaults()
        plt.ion()  # prevent memory leaks; Reverse: plt.ion()

        # Specify the color order.
        self.colorOrder = ['k', 'tab:blue', 'tab:brown', 'tab:red', 'tab:green', 'tab:purple', 'tab:pink', 'tab:orange']

    def heatmap(self, data, saveDataPath=None, title=None, xlabel=None, ylabel=None):
        # Plot the heatmap
        ax = sns.heatmap(data, robust=True, cmap='icefire')
        # Save the Figure
        sns.set(rc={'figure.figsize': (7, 9)})
        if title: ax.set_title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        fig = ax.get_figure()
        if self.saveDataFolder and saveDataPath:
            fig.savefig(f"{saveDataPath}.pdf", dpi=500, bbox_inches='tight')
            fig.savefig(f"{saveDataPath}.png", dpi=500, bbox_inches='tight')
        self.clearFigure(fig, None)
        plt.rcdefaults()

    @staticmethod
    def clearFigure(fig=None, legend=None):
        plt.show()  # Ensure the plot is displayed

        # Clear and close the figure/legend if provided
        if legend is not None: legend.remove()
        if fig: fig.clear(); plt.close(fig)

        # Clear all figures and plots
        plt.close('all')  # Close all open figures
        plt.cla()  # Clear the current axes
        plt.clf()  # Clear the current figure
        plt.rcdefaults()  # Reset Matplotlib settings to default

    @staticmethod
    def _createFolder(filePath):
        # Create the folders if they do not exist.
        os.makedirs(os.path.dirname(filePath), exist_ok=True)

    def displayFigure(self, saveFigurePath):
        if saveFigurePath is not None:
            # Create a folder for saving the file.
            self._createFolder(saveFigurePath)

            # Save the figure
            plt.savefig(saveFigurePath)
        self.clearFigure()
