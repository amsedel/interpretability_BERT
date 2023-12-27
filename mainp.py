import pygame
import sys
import numpy
from constants import *
from Point import Point
from Plot import Plot
from scale import Scale

# Inicializar pygame
pygame.init()

# Crear la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Clustering")

# Crear el cuadro de referencia
plt = Plot(PLOT_DIMS[0], PLOT_DIMS[1], PLOT_DIMS[2], PLOT_DIMS[3], 5, 5)

# Crear una serie de puntos en relación con el cuadro de referencia
puntos = [Point(numpy.array([0.7, 0.7]), plt, 0, 'uno', 0.5), Point(numpy.array([0.5, 1.1]), plt, 1, 'dos', 4.5), Point(numpy.array([0.9, 0.4]), plt, 2, 'tres', 3.5)]


def is_clicked(mouse_pos):
    x, y = mouse_pos
    for i, point in enumerate(puntos):
        px, py, pr = point.data['scale_position'][0], point.data['scale_position'][1], point.data['style']['width'] / 2
        if px - pr <= x <= px + pr and py - pr <= y <= py + pr:
            point.data['style']['color'] = RED
            print("{} , {} , {}".format(point.data['id'],point.data['sequence'],point.data['similarity']))


scale = Scale(puntos)
scale.calculate_scale()

for p in puntos:
    p.data['scale_position'] = scale.scaling(p.data['position'])

# Bucle principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Verificar si se hizo clic con el botón izquierdo del ratón
                mouse_pos = pygame.mouse.get_pos()
                point_id = is_clicked(mouse_pos)


    # Limpiar la pantalla
    screen.fill(WHITE)
    # Dibujar el cuadro de referencia
    plt.draw(screen,scale)

    # Dibujar los puntos en relación con el cuadro de referencia
    for punto in puntos:
        punto.draw(screen)

    pygame.display.flip()

# Salir del juego
pygame.quit()
sys.exit()
