from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


class plottingProtocols:

    def __init__(self, numChannels, channelDist, analysisOrder):
        matplotlib.use('Qt5Agg')  # Set Plotting GUI Backend
        # use ggplot style for more sophisticated visuals
        # plt.style.use('seaborn-poster')

        # Specify Figure aesthetics
        figWidth = 25
        figHeight = 15
        self.fig, axes = plt.subplots(numChannels, 3, sharey=False, sharex=False, gridspec_kw={'hspace': 0},
                                      figsize=(figWidth, figHeight))
        if numChannels == 1:
            axes = np.array([axes])

        self.axes = {}
        # Distribute the axes to their respective sensor
        for biomarkerInd in range(len(analysisOrder)):
            biomarkerType = analysisOrder[biomarkerInd]
            streamingChannels = channelDist[biomarkerInd]
            self.axes[biomarkerType] = axes[streamingChannels]

        # # Add experiment information to the plot
        # self.experimentText = "Current experiment: General data collection"

        # Create a surrounding figure
        self.fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        # Add figure labels
        plt.suptitle('Streaming Incoming Biolectric Signals', fontsize=22, x=0.525, fontweight="bold")
        plt.xlabel("Time (Seconds)", labelpad=15)
        # Add axis column labels
        colHeaders = ["Raw Signals", "Filtered Signals", "Selected Features"]
        for ax, colHeader in zip(axes[0], colHeaders):
            ax.set_title(colHeader, fontsize=17, pad=15)

        # Remove overlap in yTicks
        nbins = len(axes[0][0].get_yticklabels())
        for axRowInd in range(len(axes)):
            axRow = axes[axRowInd]

            for i, ax in enumerate(axRow):
                # Remove x-axis from everything except the last row
                if axRowInd != len(axes) - 1:
                    ax.set_xticks([])

                if i != 0:  # Exclude the first row
                    ax.tick_params(pad=15)  # Add padding to the left side of the subplot
                ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='both'))

        # Finalize figure spacing
        plt.tight_layout()

    def displayData(self):
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        self.fig.show()


if __name__ == "__main__":
    plottingProtocols(2, [1, 1], ['eeg', 'eog'])
