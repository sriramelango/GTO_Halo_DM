__author__ = 'rynebeeson'

from support_scripts.plot import Plot1D


# LinePlot Class is a default class for creating 2D Plots
class LinePlot(Plot1D):

    __author__ = 'rynebeeson'

    #  initialization method
    def __init__(self, axes=None, figure_number: (int, type(None)) = None):

        assert isinstance(figure_number, (int, type(None)))

        #  axis background color
        self.axisbg = '1.0'
        #  set the figure
        self.set_figure(axes, figure_number)
        #  set the axes
        if axes is None:
            self.ax = self.fig.add_subplot(111, facecolor=self.axisbg)
        else:
            self.ax = axes
            self.ax.hold(True)
        #  parent setup method to initialize basic attributes
        self.__setup__()
        #  x, y axis containers for plot scaling
        self.xaxis = []
        self.yaxis = []
        #  initialize min and max limits for setting axis
        self.limits = {'lmin': None,
                       'lmax': None,
                       'xmin': None,
                       'xmax': None,
                       'ymin': None,
                       'ymax': None}

    #  creation of a new plot using the same class object
    #  TODO: [2015-10-31] make this functionality much cleaner, plus need to update all these Plot classes
    def new_plot(self, axes=None, figure_number: (int, type(None)) = None):

        assert isinstance(figure_number, (int, type(None)))

        if self.shown or True:
            #  set the figure
            self.set_figure(axes, figure_number)
            #  set the axes
            if axes == None: self.ax  = self.fig.add_subplot(111, facecolor=self.axisbg)
            else: self.ax = axes; self.ax.hold(True)
            #  parent setup method to initialize basic attributes
            self.__setup__()
            #  reset limits
            self.reset_limits(get_new_limits = False)
        else:
            pass

    #  2d plotting method
    # TODO: [2015-11-09](plotmodule.py), colormap only works with 1d 'y' in LinePlot()
    def plot(self,
             xdata=None,
             ydata=None,
             color='blue',
             colormap=False,
             colormap_limit=1E3,
             cmap='rainbow',
             marker='',
             markersize=3.0,
             linestyle='-',
             linewidth=1.0,
             downsample=1.0,
             alpha=1.0,
             fade_alpha=False):

        # TODO: [2015-11-09](plotmodule.py), introduce asserts in plot().
        #  but must check that this is not being overdone (i.e. duplicates in future function calls)

        from numpy import arange

        assert isinstance(color, str)
        assert isinstance(colormap, bool)
        assert isinstance(alpha, float)
        assert isinstance(fade_alpha, bool)

        #  check whether we are given new data for plotting
        if not isinstance(xdata, type(None)): self.set_x(xdata)
        if not isinstance(ydata, type(None)): self.set_y(ydata)

        #  if x and y are ~2d, they may not become 2d np.ndarray because their rows are of
        #+ different length, so we must manual investigate this
        xdim = 2
        try:
            len(self.x[0])
        except TypeError:
            xdim = 1

        ydim = 2
        try:
            len(self.y[0])
        except TypeError:
            ydim = 1

        #  check whether we shouldn't generate an artificial range for the x-axis
        if len(self.x) == 0:
            if ydim == 1:
                self.set_x(range(0, len(self.y)))
            elif ydim == 2:
                new_x = []
                for y in self.y: new_x.append(arange(0, len(y)))
                self.set_x(new_x)

        #  make sure we are working with either a 1d or 2d array for 'x' and 'y'
        assert xdim in (1, 2)
        assert ydim in (1, 2)
        #  we expect the dimensions of 'y' to be the same or greater than 'x'
        assert ydim >= xdim

        #  make sure the dimensions of 'x' and 'y' match
        if xdim == 1 and ydim == 1:
            assert len(self.x) == len(self.y)
        elif xdim == 1 and ydim == 2:
            for i in range(self.y.shape[0]):
                assert len(self.x) == len(self.y[i])
        else:
            assert self.x.shape[0] == self.y.shape[0]
            for i in range(self.x.shape[0]):
                assert len(self.x[i]) == len(self.y[i])

        # plot the flow
        if ydim == 1:
            if downsample < 1.0:
                x, y = self.downsample(x=self.x, y=self.y, factor=downsample)
            else:
                x = self.x
                y = self.y

            # TODO[2015-11-09](plotmodule.py), get colormap case to work with plot using arrays for 'x', 'y' and 'color'
            # TODO[2015-11-12](LinePlot), use this class for multi-colored lines in the future:
            #  http://matplotlib.org/examples/pylab_examples/multicolored_line.html
            if colormap:
                from matplotlib.pyplot import get_cmap
                from numpy import linspace, array, concatenate

                #  check that we do not have too many points
                if len(x) > colormap_limit:
                    downsample = float(colormap_limit) / len(x)
                    x, y = self.downsample(x=x, y=y, factor=downsample)

                #  TODO: [2015-11-01], figure out what the lut arg. in get_cmap(, lut=) does, and generalizes colormaps.
                #   see: http://matplotlib.org/examples/color/colormaps_reference.html
                # cmap = get_cmap('YlGnBu', lut=None)
                cmap = get_cmap(name=cmap, lut=None)
                colors = [cmap(i) for i in linspace(0, 1, len(x))]

                #  create segments
                points = array([x, y]).T.reshape(-1, 1, 2)
                segments = concatenate([points[:-1], points[1:]], axis=1)
                colors.pop()

                for i, seg in enumerate(segments):
                    self.ax.plot(seg[:, 0], seg[:, 1],
                                 c=colors[i],
                                 marker=marker,
                                 markersize=markersize,
                                 linestyle=linestyle,
                                 linewidth=linewidth,
                                 alpha=alpha)

                # for i, color in enumerate(colors):
                #     self.ax.plot(x[i], y[i], c=color, marker=marker, markersize=markersize,
                #                  linestyle=linestyle, linewidth=linewidth)
            else:
                self.ax.plot(x, y,
                             c=color,
                             marker=marker,
                             markersize=markersize,
                             linestyle=linestyle,
                             linewidth=linewidth,
                             alpha=alpha)

            #  get the limits for the axes
            self.get_limits()
        else:
            from matplotlib.pyplot import get_cmap
            from numpy import linspace
            #  TODO: [2015-11-01], figure out what the lut arg. in get_cmap(, lut=) does, and generalizes colormaps.
            #   see: http://matplotlib.org/examples/color/colormaps_reference.html
            if colormap:
                cmap = get_cmap(cmap, lut=None)
                colors = [cmap(i) for i in linspace(0, 1, self.y.shape[0])]
            else:
                colors = self.y.shape[0]*[color]

            if xdim == 1:
                if downsample < 1.0: x = self.downsample(factor=downsample, x=self.x)
                else: x = self.x
                for i, the_y in enumerate(self.y):
                    if downsample < 1.0: y = self.downsample(factor=downsample, y=the_y)
                    else: y = the_y
                    # if fade_alpha: alpha *= (len(self.y) - i) / (len(self.y))
                    if fade_alpha: alpha *= 0.9
                    phandle, = self.ax.plot(x, y,
                                            c=colors[i],
                                            marker=marker,
                                            markersize=markersize,
                                            linestyle=linestyle,
                                            linewidth=linewidth,
                                            alpha=alpha)

                    #  get the limits for the axes
                    self.get_limits(x=x, y=y)
            else:
                for i, the_y in enumerate(self.y):
                    if downsample < 1.0:
                        x, y = self.downsample(factor=downsample, x=self.x[i], y=the_y)
                    else:
                        x = self.x[i]
                        y = the_y
                    phandle, = self.ax.plot(x, y,
                                            c=colors[i],
                                            marker=marker,
                                            markersize=markersize,
                                            linestyle=linestyle,
                                            linewidth=linewidth,
                                            alpha=alpha)

                    #  get the limits for the axes
                    self.get_limits(x=x, y=y)
            return phandle

        """
        #  find the min and max of the plot data
        lmins = [min(self.x), min(self.y)]
        lmaxs = [max(self.x), max(self.y)]
        lmin  = min(lmins)
        lmax  = max(lmaxs)
        #  find the extreme min and max of the plot data
        #+ and save to self attributes if
        #+ max is larger than current self.max or
        #+ min is smaller than current self.min
        if lmin < self.lmin: self.lmin = lmin
        if lmax > self.lmax: self.lmax = lmax
        """

    #  set the axes
    def set_axis(self, aspect='equal', limits=None,
                 ybuffer=None, xbuffer=None):

        assert aspect in ['equal', 'tight']
        assert isinstance(limits, (type(None), list))

        #  if the user doesn't specify the limits
        if limits is None:
            #  if aspect is set to equal by default or user
            if aspect == 'equal':
                limits = [self.limits['lmin'], self.limits['lmax'],
                          self.limits['lmin'], self.limits['lmax']]
            elif aspect == 'tight':
                limits = [self.limits['xmin'], self.limits['xmax'],
                          self.limits['ymin'], self.limits['ymax']]
        else:
            assert len(limits) == 4
            if limits[0] is None: limits[0] = self.limits['xmin']
            if limits[1] is None: limits[1] = self.limits['xmax']
            if limits[2] is None: limits[2] = self.limits['ymin']
            if limits[3] is None: limits[3] = self.limits['ymax']

        if ybuffer is not None:
            ydiff = ybuffer * (limits[3] - limits[2])
            limits[3] += ydiff
            limits[2] -= ydiff
        if xbuffer is not None:
            xdiff = xbuffer * (limits[1] - limits[0])
            limits[1] += xdiff
            limits[0] -= xdiff

        #  set the limits
        if (limits[1] - limits[0]) > 1E-6: self.ax.set_xlim(limits[0], limits[1])
        if (limits[3] - limits[2]) > 1E-6: self.ax.set_ylim(limits[2], limits[3])

    #  get the axes limits
    def get_limits(self, x = None, y = None):
        #  find the min and max of the plot data
        if isinstance(x, type(None)): x = self.x
        if isinstance(y, type(None)): y = self.y

        xmin = min(x); xmax = max(x)
        ymin = min(y); ymax = max(y)
        lmins = [xmin, ymin]
        lmaxs = [xmax, ymax]
        lmin  = min(lmins)
        lmax  = max(lmaxs)
        #  find the extreme min and max of the plot data
        #+ and save to the dictionary if
        #+ max is larger than current max or
        #+ min is smaller than current min
        #  lmin
        if self.limits['lmin'] == None:
            self.limits['lmin'] = lmin
        elif lmin < self.limits['lmin']:
            self.limits['lmin'] = lmin
        #  lmax
        if self.limits['lmax'] == None:
            self.limits['lmax'] = lmax
        elif lmax > self.limits['lmax']:
            self.limits['lmax'] = lmax
        #  xmin
        if self.limits['xmin'] == None:
            self.limits['xmin'] = xmin
        elif xmin < self.limits['xmin']:
            self.limits['xmin'] = xmin
        #  xmax
        if self.limits['xmax'] == None:
            self.limits['xmax'] = xmax
        elif xmax > self.limits['xmax']:
            self.limits['xmax'] = xmax
        #  ymin
        if self.limits['ymin'] == None:
            self.limits['ymin'] = ymin
        elif ymin < self.limits['ymin']:
            self.limits['ymin'] = ymin
        #  ymax
        if self.limits['ymax'] == None:
            self.limits['ymax'] = ymax
        elif ymax > self.limits['ymax']:
            self.limits['ymax'] = ymax

    #  reset the axes limits with the current data
    def reset_limits(self, get_new_limits = True):
        self.limits = {'lmin': None,
                        'lmax': None,
                        'xmin': None,
                        'xmax': None,
                        'ymin': None,
                        'ymax': None}
        if get_new_limits:
            self.get_limits()

    #  set the aspect ratio
    def set_aspect(self, aspect='equal'):
        if aspect == 'equal':
            self.ax.set_aspect(aspect)
        else:
            self.ax.set_aspect(aspect)