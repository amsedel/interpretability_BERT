import numpy as np
import math
from constants import *

class Scale:
    def __init__(self, points) -> None:
        self.points = points
        self.xy_scale = 1
        self.canvas_dims = np.array([[PLOT_DIMS[0],PLOT_DIMS[1]],[PLOT_DIMS[2],PLOT_DIMS[3]]])
        self.canvas_origin = np.array([PLOT_DIMS[0],self.canvas_dims[1][0]])

    def calculate_dims(self):
        positions = np.array([p.data['position'] for p in self.points])
        # Calcular el valor máximo y mínimo de las coordenadas
        self.max_value = np.max(positions, axis=0)
        self.min_value = np.min(positions, axis=0)
        return np.array([self.min_value,self.max_value])


    def calculate_scale(self):
        self.dims = self.calculate_dims()
        self.plot_size = self.dims[1] - self.dims[0]
        self.canvas_size = self.canvas_dims[1]-self.canvas_dims[0]
        #Calculate scales
        x_scale = self.canvas_size[0] / self.plot_size[0]
        y_scale = self.canvas_size[1] / self.plot_size[1]
        if y_scale > x_scale:
            displacement = [0, ((y_scale - x_scale) / (y_scale * 2)) * self.canvas_size[1]]
            self.xy_scale = x_scale
        else: 
            displacement = [((x_scale - y_scale) / (x_scale * 2)) * self.canvas_size[0], 0]
            self.xy_scale = y_scale
        self.gap = np.array(displacement)

    def scaling(self, point_pos):
        pos = (self.xy_scale * (point_pos-self.dims[0]))
        pos_canvas = self.canvas_origin + self.gap 
        return  np.array([pos_canvas[0]+pos[0], pos_canvas[1]-pos[1]])