import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Button

"""
matplotlib GUI for creating x-y plots of dictionary with dropdown menu options for axes.
Matplotlib does not support dropdown menu, so this is based on buttons for now.

Todo: incorate as subclass of MeasurementSelector and add ability to click-and-drag
and interval, so that experiment.explore() can be used to also select a region into
a class attribute.
"""

class ButtonPlotter:
    """
    matplotlib GUI for creating x-y plots of dictionary with dropdown menu options for axes.
    Matplotlib does not support dropdown menu, so this is based on buttons for now.

    Todo: incorate as subclass of MeasurementSelector and add ability to click-and-drag
    and interval, so that experiment.explore() can be used to also select a region into
    a class attribute.
    """

    def __init__(self, headers, data):

        self.headers = headers
        self.data = data
        self.num_headers = len(headers)

        # set figure dimensions
        self.figx = 10
        self.figy = 5

        # set plot and button dimensions, registered by bottom left corner and
        # as fractions of figure dims. margin sets spacing between objects.
        margin = 0.01
        self.ax_x = 0.5
        self.ax_y = 0.2 + margin
        self.ax_width = 0.5 - margin
        self.ax_height = 0.8 - 2*margin
        self.buttons_x = margin
        self.buttons_y = margin
        self.buttons_width = 0.2 - 2*margin
        self.buttons_height = 1 - 2*margin
        self.stop_width = .2
        self.stop_height = .05
        self.stop_x = 0.75 - self.stop_width/2
        self.stop_y = 0.05

        # set default x and y axes
        self.x = headers[0]
        self.y = headers[1]
        self.xdata = data[self.x]
        self.ydata = data[self.y]
        self.xlims = (np.nanmin(self.xdata), np.nanmax(self.xdata))
        self.ylims = (np.nanmin(self.ydata), np.nanmax(self.ydata))

    def execute(self):

        # initialize plot
        self.fig, self.ax = plt.subplots(figsize=(self.figx, self.figy))
        self.ax.set_position(pos=[self.ax_x, self.ax_y, self.ax_width, self.ax_height])
        #plt.subplots_adjust(left=self.figy/2, bottom=self.figy)
        self.p = self.ax.plot(self.xdata, self.ydata, 'o', ms=0.1)
        self.ax.set_xlabel(self.x)
        self.ax.set_ylabel(self.y)
        self.ax.set_xlim(*self.xlims)
        self.ax.set_ylim(*self.ylims)

        # event handlers for clicks
        def set_x(attribute):
            self.xdata = self.data[attribute]
            self.p[0].set_data(self.xdata, self.ydata)
            self.ax.set_xlabel(attribute)
            self.ax.set_xlim(np.nanmin(self.xdata), np.nanmax(self.xdata))
            self.fig.canvas.draw_idle()

        def set_y(attribute):
            self.ydata = self.data[attribute]
            self.p[0].set_data(self.xdata, self.ydata)
            self.ax.set_ylabel(attribute)
            self.ax.set_ylim(np.nanmin(self.ydata), np.nanmax(self.ydata))
            self.fig.canvas.draw_idle()

        def stop(val):
            self.x_button.disconnect(self.cid_x)
            self.y_button.disconnect(self.cid_y)
            self.s_button.disconnect(self.cid_s)
            plt.close(self.fig)
            print('closed figure')

        # define dimensions of buttons
        self.ax_xbutton = plt.axes([self.buttons_x, self.buttons_y, self.buttons_width, self.buttons_height], figure=self.fig)
        self.ax_ybutton = plt.axes([self.buttons_x+self.buttons_width, self.buttons_y, self.buttons_width, self.buttons_height],figure=self.fig)
        self.ax_sbutton = plt.axes([self.stop_x, self.stop_y, self.stop_width, self.stop_height],figure=self.fig)

        # define buttons
        self.x_button = RadioButtons(self.ax_xbutton, self.headers, active=0)
        self.y_button = RadioButtons(self.ax_ybutton, self.headers, active=0)
        self.s_button = Button(self.ax_sbutton, 'Stop')
        self.x_button.set_active(0)
        self.y_button.set_active(1)

        # initiate behavior of buttons when clicked
        self.cid_x = self.x_button.on_clicked(set_x)
        self.cid_y = self.y_button.on_clicked(set_y)
        self.cid_s = self.s_button.on_clicked(stop)

        plt.show()
