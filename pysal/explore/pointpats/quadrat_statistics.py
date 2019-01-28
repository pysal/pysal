"""
Quadrat statistics for planar point patterns


TODO

- use patch in matplotlib to plot rectangles and hexagons
- plot chi2 statistics in each cell
- delete those cells that do not intersect with the window (study area)

"""

__author__ = 'Serge Rey, Wei Kang, Hu Shao'
__all__ = ['RectangleM', 'HexagonM', 'QStatistic']

import numpy as np
from matplotlib import pyplot as plt
import math
import scipy

class RectangleM:
    """
    Rectangle grid structure for quadrat-based method.

    Parameters
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    count_column      : integer
                        Number of rectangles in the horizontal
                        direction. Use in pair with count_row to
                        fully specify a rectangle. Incompatible with
                        rectangle_width and rectangle_height.
    count_row         : integer
                        Number of rectangles in the vertical
                        direction. Use in pair with count_column to
                        fully specify a rectangle. Incompatible with
                        rectangle_width and rectangle_height.
    rectangle_width   : float
                        Rectangle width. Use in pair with
                        rectangle_height to fully specify a rectangle.
                        Incompatible with count_column & count_row.
    rectangle_height  : float
                        Rectangle height. Use in pair with
                        rectangle_width to fully specify a rectangle.
                        Incompatible with count_column & count_row.

    Attributes
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    mbb               : array
                        Minimum bounding box for the point pattern.
    points            : array
                        x,y coordinates of the point points.
    count_column      : integer
                        Number of columns.
    count_row         : integer
                        Number of rows.
    num               : integer
                        Number of rectangular quadrats.

    """

    def __init__(self, pp, count_column = 3, count_row = 3,
                 rectangle_width = 0, rectangle_height = 0):
        self.mbb = pp.mbb
        self.pp = pp
        self.points = np.asarray(pp.points)
        x_range = self.mbb[2]-self.mbb[0]
        y_range = self.mbb[3]-self.mbb[1]
        if rectangle_width & rectangle_height:
            self.rectangle_width = rectangle_width
            self.rectangle_height = rectangle_height

            # calculate column count and row count
            self.count_column = int(math.ceil(x_range / rectangle_width))
            self.count_row = int(math.ceil(y_range / rectangle_height))
        else:
            self.count_column = count_column
            self.count_row = count_row

            # calculate the actual width and height of cell
            self.rectangle_width = x_range/float(count_column)
            self.rectangle_height = y_range/float(count_row)
        self.num = self.count_column * self.count_row

    def point_location_sta(self):
        """
        Count the point events in each cell.

        Returns
        -------
        dict_id_count : dict
                        keys: rectangle id, values: number of point
                        events in each cell.
        """

        dict_id_count = {}
        for i in range(self.count_row):
            for j in range(self.count_column):
                dict_id_count[j+i*self.count_column] = 0

        for point in self.points:
            index_x = (point[0]-self.mbb[0]) // self.rectangle_width
            index_y = (point[1]-self.mbb[1]) // self.rectangle_height
            if index_x == self.count_column:
                index_x -= 1
            if index_y == self.count_row:
                index_y -= 1
            id = index_y * self.count_column + index_x
            dict_id_count[id] += 1
        return dict_id_count

    def plot(self, title="Quadrat Count"):
        '''
        Plot rectangle tessellation as well as the number of points falling in each rectangle.

        Parameters
        ----------
        title:   str, optional
                 Title of the plot. Default is "Quadrat Count".

        '''

        line_width_cell = 1
        line_color_cell = 'red'

        x_min = self.mbb[0]
        y_min = self.mbb[1]

        # draw the point pattern along with its window
        ax = self.pp.plot(window=True, title=title,
                         get_ax=True)

        # draw cells and counts
        x_start_end = [x_min,
                       x_min + self.count_column*self.rectangle_width]
        for row in range(self.count_row + 1):
            y = y_min + row*self.rectangle_height
            ax.plot(x_start_end, [y, y], lw = line_width_cell,
                    color=line_color_cell)
        y_start_end = [y_min,
                       y_min + self.count_row*self.rectangle_height]
        for column in range(self.count_column + 1):
            x = x_min + column*self.rectangle_width
            ax.plot([x, x], y_start_end, lw = line_width_cell,
                    color=line_color_cell)

        dict_id_count = self.point_location_sta()
        for x in range(self.count_column):
            for y in range(self.count_row):
                cell_id = x + y*self.count_column
                count = dict_id_count[cell_id]
                position_x = x_min + self.rectangle_width*(x+0.5)
                position_y = y_min + self.rectangle_height*(y+0.5)
                ax.text(position_x, position_y, str(count))
        plt.show()


class HexagonM:
    """
    Hexagon grid structure for quadrat-based method.

    Parameters
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    lh                : float
                        Hexagon length (hexagon).

    Attributes
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    h_length          : float
                        Hexagon length (hexagon).
    mbb               : array
                        Minimum bounding box for the point pattern.
    points            : array
                        x,y coordinates of the point points.
    h_length          : float
                        Hexagon length (hexagon).
    count_row_even    : integer
                        Number of even rows.
    count_row_odd     : integer
                        Number of odd rows.
    count_column      : integer
                        Number of columns.
    num               : integer
                        Number of hexagonal quadrats.

    """
    def __init__(self, pp, lh):

        self.points = np.asarray(pp.points)
        self.pp = pp
        self.h_length = lh
        self.mbb = pp.mbb
        range_x = self.mbb[2] - self.mbb[0]
        range_y = self.mbb[3] - self.mbb[1]

        # calculate column count
        self.count_column = 1
        if self.h_length/2.0 < range_x:
            temp = math.ceil((range_x - self.h_length/2) / (
                1.5 * self.h_length))
            self.count_column += int(temp)

        # calculate row count for the even columns
        self.semi_height = self.h_length * math.cos(math.pi/6)
        self.count_row_even = 1
        if self.semi_height < range_y:
            temp = math.ceil((range_y-self.semi_height)/(
                self.semi_height*2))
            self.count_row_even += int(temp)

        # for the odd columns
        self.count_row_odd = int(math.ceil(range_y/(self.semi_height*2)))

        # quadrat number
        self.num = self.count_row_odd * ((self.count_column // 2) +
                                         self.count_column % 2) + \
                   self.count_row_even * (self.count_column // 2)

    def point_location_sta(self):
        """
        Count the point events in each hexagon cell.

        Returns
        -------
        dict_id_count : dict
                        keys: rectangle id, values: number of point
                        events in each hexagon cell.
        """
        semi_cell_length = self.h_length / 2.0
        dict_id_count = {}

        # even row may be equal with odd row or 1 more than odd row
        for i in range(self.count_row_even):
            for j in range(self.count_column):
                if self.count_row_even != self.count_row_odd and i ==\
                                self.count_row_even-1:
                    if j % 2 == 1:
                        continue
                dict_id_count[j+i*self.count_column] = 0

        x_min = self.mbb[0]
        y_min = self.mbb[1]
        x_max = self.mbb[2]
        y_max = self.mbb[3]
        points = np.array(self.points)
        for point in points:
            # find the possible x index
            intercept_degree_x = ((point[0]-x_min)//semi_cell_length)

            # find the possible y index
            possible_y_index_even = int((point[1]+ self.semi_height -
                                         y_min)/ (self.semi_height * 2))
            possible_y_index_odd = int((point[1] - y_min) / (
                self.semi_height * 2))
            if intercept_degree_x % 3 != 1:
                center_index_x = (intercept_degree_x+1) // 3
                center_index_y = possible_y_index_odd
                if center_index_x % 2 == 0:
                    center_index_y = possible_y_index_even
                dict_id_count[center_index_x + center_index_y * self.count_column] += 1
            else: # two columns of cells can be possible
                center_index_x = intercept_degree_x//3
                center_x = center_index_x*semi_cell_length*3 + x_min
                center_index_y = possible_y_index_odd
                center_y = (center_index_y*2+1)*self.semi_height + y_min
                if center_index_x % 2 == 0:
                    center_index_y = possible_y_index_even
                    center_y = center_index_y*self.semi_height*2 + y_min

                if point[1] > center_y:  # compare the upper bound
                    x0 = center_x+self.h_length
                    y0 = center_y
                    x1 = center_x+semi_cell_length
                    y1 = center_y+self.semi_height
                    indicator = -(point[1] - ((y0-y1)/(x0-x1)*point[
                        0] + (x0*y1-x1*y0)/(x0-x1)))
                else:  #compare the lower bound
                    x0 = center_x+semi_cell_length
                    y0 = center_y-self.semi_height
                    x1 = center_x+self.h_length
                    y1 = center_y
                    indicator = point[1] - ((y0-y1)/(x0-x1)*point[0]
                                            + (x0*y1-x1*y0)/(x0-x1))
                if indicator <= 0:
                    # we select right hexagon instead of the left
                    center_index_x += 1
                    center_index_y = possible_y_index_odd
                    if center_index_x % 2 == 0:
                        center_index_y = possible_y_index_even
                dict_id_count[center_index_x + center_index_y
                              * self.count_column] += 1
        return dict_id_count

    def plot(self, title="Quadrat Count"):
        '''
        Plot hexagon quadrats as well as the number of points falling in each quadrat.

        Parameters
        ----------
        title:   str, optional
                 Title of the plot. Default is "Quadrat Count".

        '''
        line_width_cell = 1
        line_color_cell = 'red'

        # draw the point pattern along with its window
        ax = self.pp.plot(window=True, title= title,
                         get_ax=True)

        x_min = self.mbb[0]
        y_min = self.mbb[1]

        # draw cells and counts
        dict_id_count = self.point_location_sta()
        for id in dict_id_count.keys():
            index_x = id % self.count_column
            index_y = id // self.count_column
            center_x = index_x*self.h_length/2.0*3.0 + x_min
            center_y = index_y*self.semi_height*2.0 + y_min
            if index_x % 2 == 1:  # for the odd columns
                center_y = (index_y*2.0+1)*self.semi_height + y_min
            list_points_cell = []
            list_points_cell.append([center_x + self.h_length,
                                     center_y])
            list_points_cell.append([center_x
                                     + self.h_length/2,
                                     center_y + self.semi_height])
            list_points_cell.append([center_x
                                     - self.h_length/2,
                                     center_y+self.semi_height])
            list_points_cell.append([center_x - self.h_length,
                                     center_y])
            list_points_cell.append([center_x
                                     - self.h_length/2,
                                     center_y-self.semi_height])
            list_points_cell.append([center_x
                                     + self.h_length/2,
                                     center_y-self.semi_height])
            list_points_cell.append([center_x + self.h_length,
                                     center_y])
            ax.plot(np.array( list_points_cell)[:,0],np.array(
                    list_points_cell)[:,1], lw =line_width_cell,
                    color=line_color_cell)

            ax.text(center_x, center_y, str(dict_id_count[id]))
        plt.show()



class QStatistic:
    """
    Quadrat analysis of point pattern.

    Parameters
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    shape             : string
                        Grid structure. Either "rectangle" or "hexagon".
                        Default is "rectangle".
    nx                : integer
                        Number of rectangles in the horizontal
                        direction. Only when shape is specified as
                        "rectangle" will nx be considered.
    ny                : integer
                        Number of rectangles in the vertical direction.
                        Only when shape is specified as "rectangle"
                        will ny be considered.
    lh                : float
                        Hexagon length (hexagon). Only when shape is
                        specified as "hexagon" will lh be considered.
                        Incompatible with nx & ny.
    realizations      : :class:`PointProcess`
                        Point process instance with more than 1 point
                        pattern realizations which would be used for
                        simulation based inference. Default is 0
                        where no simulation based inference is
                        performed.
                        
    Attributes
    ----------
    pp                : :class:`.PointPattern`
                        Point Pattern instance.
    mr                : :class:`.RectangleM` or :class:`.HexagonM`
                        RectangleM or HexagonM instance.
    chi2              : float
                        Chi-squared test statistic for the observed
                        point pattern pp.
    df                : integer
                        Degree of freedom.
    chi2_pvalue       : float
                        p-value based on analytical chi-squared
                        distribution.
    chi2_r_pvalue     : float
                        p-value based on simulated sampling
                        distribution. Only available when
                        realizations is correctly specified.
    chi2_realizations : array
                        Chi-squared test statistics calculated for
                        all of the simulated csr point patterns.
    """

    def __init__(self, pp, shape= "rectangle",nx = 3, ny = 3,
                 lh = 10, realizations = 0):
        self.pp = pp
        if shape == "rectangle":
            self.mr = RectangleM(pp, count_column = nx,
                                        count_row = ny)
        elif shape == "hexagon":
            self.mr = HexagonM(pp,lh)

        # calculate chi2 test statisitc for the observed point pattern
        dict_id_count = self.mr.point_location_sta()
        self.chi2,self.chi2_pvalue = scipy.stats.chisquare(
                list(dict_id_count.values()))

        self.df = self.mr.num - 1

        # when realizations is specified, perform simulation based
        # inference.
        if realizations:
            reals = realizations.realizations
            sim_n = realizations.samples
            chi2_realizations = [] #store test statisitcs for all the

            for i in range(sim_n):
                if shape == "rectangle":
                    mr_temp = RectangleM(reals[i],
                                                count_column=nx,
                                                count_row=ny)
                elif shape == "hexagon":
                    mr_temp = HexagonM(reals[i],lh)
                id_count_temp = mr_temp.point_location_sta().values()

                #calculate test statistics for simulated point patterns
                chi2_sim,p = scipy.stats.chisquare(list(id_count_temp))
                chi2_realizations.append(chi2_sim)
            self.chi2_realizations = np.array(chi2_realizations)

            #calculate pseudo pvalue
            above_chi2 = self.chi2_realizations >= self.chi2
            larger_chi2 = sum(above_chi2)
            self.chi2_r_pvalue = (larger_chi2 + 1.)/(sim_n+ 1.)

    def plot(self, title = "Quadrat Count"):
        '''
        Plot quadrats as well as the number of points falling in each quadrat.

        Parameters
        ----------
        title:   str, optional
                 Title of the plot. Default is "Quadrat Count".

        '''

        self.mr.plot(title = title)
