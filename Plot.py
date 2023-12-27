import pygame
from constants import *
import numpy as np

# Clase para el cuadro de referencia
class Plot:
    def __init__(self, x, y, width, height, divisions_x = 10, divisions_y = 10):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.divisions_x = divisions_x
        self.divisions_y = divisions_y
        self.fontsize = FONTSIZE

    def draw(self, screen, scale):
        #pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)
        pygame.draw.line(screen, BLACK, (self.x, self.y), (self.x, self.height), 3)
        pygame.draw.line(screen, BLACK, (self.x, self.height), (self.width, self.height), 3)

        # Etiquetas de ejes y números
        font = pygame.font.Font(None, self.fontsize)
        x_label = font.render("X", True, BLACK)
        y_label = font.render("Y", True, BLACK)
        screen.blit(x_label, (PLOT_DIMS[2]+self.fontsize/2, PLOT_DIMS[3]+self.fontsize/2))
        screen.blit(y_label, (PLOT_DIMS[0]-self.fontsize, PLOT_DIMS[1]-self.fontsize))

        # Dibujar números en los ejes
        gap_x = scale.min_value[0]
        delta_x = (scale.max_value[0] - scale.min_value[0]) / (self.divisions_x - 1)
        for i in range(self.divisions_x):
            x_value = font.render(str(round(gap_x,2)), True, BLACK)
            pos = np.array([gap_x, 0])
            x_pos = scale.scaling(pos)
            screen.blit(x_value, np.array([x_pos[0], self.height + 10]))
            gap_x = gap_x + delta_x

        gap_y = scale.min_value[1]
        delta_y = (scale.max_value[1] - scale.min_value[1]) / (self.divisions_y - 1)
        for j in range(self.divisions_y):
            y_value = font.render(str(round(gap_y,2)), True, BLACK)
            pos = np.array([0, gap_y])
            y_pos = scale.scaling(pos)
            screen.blit(y_value, np.array([self.x-40, y_pos[1]]))
            gap_y = gap_y + delta_y