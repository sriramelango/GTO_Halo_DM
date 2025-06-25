class Plot:
    """
    Plot Class is a general default class for creating child plot classes
    """

    __author__ = 'rynebeeson'

    #  set the figure
    def set_figure(self, axes, figure_number=None):

        assert isinstance(figure_number, (int, type(None)))

        #  import statements
        import matplotlib.pyplot as plt
        if axes is None:
            if figure_number is None:
                self.fig = plt.figure()
            else:
                self.fig = plt.figure(figure_number)
        else:
            self.fig = plt.figure(axes.figure.number)

        #  FIXME: i don't believe this plt.get_fignums() is local; it should generate a list of all fignums()
        # self.figure_number = plt.get_fignums()[0]
        self.shown = False

    #  FUTUREWORK, use artist for proper control of legend
    #  set the legend
    #  see legend API for more options: http://matplotlib.org/api/legend_api.html
    def set_legend(self, legend_list=[''], location=0, artist: (tuple, type(None))=None):
        self.ax.legend(legend_list, loc=location)

    #  draw the plot
    def draw(self):
        #  import statements
        import matplotlib.pyplot as plt
        plt.draw()

    #  draw a grid on plot
    def grid(self):
        #  import statements
        import matplotlib.pyplot as plt
        plt.grid()

    #  show the plot
    def show(self):
        #  import statements
        import matplotlib.pyplot as plt
        # plt.show(self.fig.number)
        plt.show()
        self.shown = True

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig.number)

    #  save the plot
    def save_figure(self, full_filepath="~/Desktop/Plot.png", dpi=100):
        assert isinstance(full_filepath, str)
        from matplotlib.pylab import savefig
        #savefig(full_filepath, format=full_filepath[-3:], dpi=dpi)
        savefig(full_filepath, dpi=dpi)

    def downsample(self, factor, x):
        return x[0::int(1 / factor)]

    #  method for setting the x data
    def set_x(self, x):
        """
           'vararg' can be given as a number of types, which all trigger
           different setup behavior.
           if given as a list or numpy array then this is assigned to 'x'
           if given as a scalar and 'ub', 'delta' arguements are not None, then
            a numpy array is created using 'vararg' as the 'lb'
           if given as a function handle, then the function is called with no
            input arguments
        """
        from numpy import ndarray, array
        assert isinstance(x, (list, ndarray)) or callable(x)

        if isinstance(x, list): x = array(x)
        if isinstance(x, ndarray): self.x = x
        else: self.x = x()


class Plot1D(Plot):
    """
    Plot1D Class is a general default class for creating child 1D plot classes
    of 1D Manifold Objects Embedded in 2D Space
    """

    #  basic member setup method for child classes
    def __setup__(self):
        #  additional plotting attributes
        self.xlabel = ''
        self.ylabel = ''
        self.title  = ''
        #  x, y containers for plotting
        self.x = []
        self.y = []

    #  set and print title
    def set_title(self, title_in, fontsize=12):
        import matplotlib.pyplot as plt
        self.title = title_in
        plt.title(title_in, fontsize=fontsize)

    #  set and print xlabel
    def set_xlabel(self, xlabel_in, fontsize=12):
        assert isinstance(xlabel_in, str)
        assert isinstance(fontsize, int)
        import matplotlib.pyplot as plt
        self.xlabel = xlabel_in
        plt.xlabel(xlabel_in, fontsize=fontsize)

    #  set and print ylabel
    def set_ylabel(self, ylabel_in, fontsize=12):
        assert isinstance(ylabel_in, str)
        assert isinstance(fontsize, int)
        import matplotlib.pyplot as plt
        self.ylabel = ylabel_in
        plt.ylabel(ylabel_in, fontsize=fontsize)

    #  method for setting the y data
    def set_y(self, y):
        """
            'vararg' can be given as a number of types, which all trigger
            different setup behavior.
            if given as a list or numpy array then this is assigned to 'y'
            if given as a function handle, then the function is called and will
             pass self.x as an input argument
        """
        from numpy import ndarray, array
        assert isinstance(y, (list, ndarray)) or callable(y)

        if isinstance(y, list): y = array(y)
        if isinstance(y, ndarray): self.y = y
        else: self.y = y(self.x)

    def downsample(self, factor, x=None, y=None):

        xNone = isinstance(x, type(None))
        yNone = isinstance(y, type(None))

        assert not xNone or not yNone

        if not xNone and not yNone:
            return x[0::int(1 / factor)], y[0::int(1 / factor)]
        if not yNone:
            return y[0::int(1 / factor)]
        else:
            return x[0::int(1 / factor)]


class Plot2D(Plot):
    """
    Plot2D Class is a general default class for creating child 2D plot classes
    of 2D Manifold Objects Embedded in 2D OR 3D Space
    """

    #  basic member setup method for child classes
    def __setup__(self):
        #  additional plotting attributes
        self.xlabel = ''
        self.ylabel = ''
        self.title  = ''
        #  x, y containers for plotting
        self.x = []
        self.y = []

    #  method for setting the y data
    def set_y(self, y):
        """
            'vararg' can be given as a number of types, which all trigger
            different setup behavior.
            if given as a list or numpy array then this is assigned to 'y'
            if given as a scalar and 'ub', 'delta' arguements are not None, then
             a numpy array is created using 'vararg' as the 'lb'
            if given as a function handle, then the function is called and will
             pass self.x as an input argument
            """
        from numpy import ndarray, array
        assert isinstance(y, (list, ndarray)) or callable(y)

        if isinstance(y, list): y = array(y)
        if isinstance(y, ndarray): self.y = y
        else: self.y = y(self.x)


    #  method for creating the x, y grid
    def set_grid(self, x=None, y=None):
        #  import statements
        from numpy import meshgrid
        if isinstance(x, type(None)) and isinstance(y, type(None)):
            self.x, self.y = meshgrid(self.x, self.y)
        else:
            self.x, self.y = meshgrid(x, y)

    #  method for setting the y data
    def set_z(self, z):
        """
            'vararg' can be given as a number of types, which all trigger
            different setup behavior.
            if given as a 2d list or 2d numpy array then this is assigned to 'z'
            if given as a function handle, then the function is called and will
            pass self.x and self.y as input arguments
        """

        from numpy import ndarray, array
        assert isinstance(z, (list, ndarray)) or callable(z)


        #  import
        from copy import copy
        #  setup the x, y grid
        self.set_grid()


        if isinstance(z, list): z = array(z)
        if isinstance(z, ndarray): self.z = z
        #  TODO: [2015-11-01] check that using the z-function set method works...
        else:
            self.z = copy(self.x)
            #  compute the z values based on the given function handle
            for i in range(self.x.shape[0]):
                for j in range(self.x.shape[1]):
                    self.z[i][j] = z([self.x[i][j], self.y[i][j]])

    def downsample(self, factor, x=None, y=None, z=None):

        xSample = not isinstance(x, type(None))
        ySample = not isinstance(y, type(None))
        zSample = not isinstance(z, type(None))

        xyzSample = (xSample, ySample, zSample)

        assert any(xyzSample)

        if all(xyzSample):
            return x[0::int(1 / factor)], y[0::int(1 / factor)], z[0::int(1 / factor)]
        elif all((xSample, ySample)):
            return x[0::int(1 / factor)], y[0::int(1 / factor)]
        elif all((xSample, zSample)):
            return x[0::int(1 / factor)], z[0::int(1 / factor)]
        elif all((ySample, zSample)):
            return y[0::int(1 / factor)], z[0::int(1 / factor)]
        elif xSample:
            return x[0::int(1 / factor)]
        elif ySample:
            return y[0::int(1 / factor)]
        else:
            return z[0::int(1 / factor)]


class Plot3D(Plot):
    """
    Plot3D Class is a general default class for creating child 3D plot classes of
    """

    #  basic member setup method for child classes
    def __setup__(self):
        #  additional plotting attributes
        self.xlabel = ''
        self.ylabel = ''
        self.zlabel = ''
        self.title  = ''
        #  x, y, z containers for plotting
        self.x = []
        self.y = []
        self.z = []

    # TODO: [2015-11-12](plotmodule.py), this is currently a duplicate of that in Plot2D
    def downsample(self, factor, x=None, y=None, z=None):

        xSample = not isinstance(x, type(None))
        ySample = not isinstance(y, type(None))
        zSample = not isinstance(z, type(None))

        xyzSample = (xSample, ySample, zSample)

        assert any(xyzSample)

        if all(xyzSample):
            return x[0::int(1 / factor)], y[0::int(1 / factor)], z[0::int(1 / factor)]
        elif all((xSample, ySample)):
            return x[0::int(1 / factor)], y[0::int(1 / factor)]
        elif all((xSample, zSample)):
            return x[0::int(1 / factor)], z[0::int(1 / factor)]
        elif all((ySample, zSample)):
            return y[0::int(1 / factor)], z[0::int(1 / factor)]
        elif xSample:
            return x[0::int(1 / factor)]
        elif ySample:
            return y[0::int(1 / factor)]
        else:
            return z[0::int(1 / factor)]

    #  method for setting the y data
    def set_y(self, y):
        """
            if given as a list or numpy array then this is assigned to 'y'
            """
        from numpy import ndarray, array
        assert isinstance(y, (list, ndarray))

        if isinstance(y, list): y = array(y)
        if isinstance(y, ndarray): self.y = y

    #  method for setting the z data
    def set_z(self, z):
        """
            if given as a list or numpy array then this is assigned to 'y'
            """
        from numpy import ndarray, array
        assert isinstance(z, (list, ndarray))

        if isinstance(z, list): z = array(z)
        if isinstance(z, ndarray): self.z = z

    def set_xlabel(self, label='x', fontsize=12):
        assert isinstance(label, str)
        assert isinstance(fontsize, int)
        self.ax.set_xlabel(label, fontsize=fontsize)

    def set_ylabel(self, label='y', fontsize=12):
        assert isinstance(label, str)
        assert isinstance(fontsize, int)
        self.ax.set_ylabel(label, fontsize=fontsize)

    def set_zlabel(self, label='z', fontsize=12):
        assert isinstance(label, str)
        assert isinstance(fontsize, int)
        self.ax.set_zlabel(label, fontsize=fontsize)

    def view(self, elevation=45.0, azimuth=45.0):
        assert isinstance(elevation, float)
        assert isinstance(azimuth, float)
        self.ax.view_init(elev=elevation, azim=azimuth)
