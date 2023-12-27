import pygame
from constants import *
import numpy

# Clase para los puntos
class Point:
    def __init__(self, position, reference_square, id, text, similarity, color = BLUE, width = POINT_WIDTH):
        self.x = position[0]
        self.y = position[1]
        self.reference = reference_square
        self.data = {
            'sequence': text,
            'similarity': similarity,
            'id': id,
            'position': numpy.array(position),
            'scale_position': numpy.array([0, 0]),
            'style': {
                'width': width,
                'color': color
            }
        }

    def draw(self, canvas):
        #pygame.draw.circle(canvas, self.data['style']['color'], self.data['position'], self.data['style']['width'])
        #pygame.draw.circle(canvas, self.data['style']['color'], (self.reference.x + self.x, self.reference.y + self.y), self.data['style']['width'])
        pygame.draw.circle(canvas, self.data['style']['color'], self.data['scale_position'], self.data['style']['width'])