import pygame
import sys

# Inicializar pygame
pygame.init()

# Dimensiones de la ventana
WIDTH, HEIGHT = 800, 600

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)

# Clase para el cuadro de referencia
class CuadroReferencia:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def dibujar(self, screen):
        pygame.draw.rect(screen, BLACK, (self.x, self.y, self.width, self.height), 2)

# Clase para los puntos
class Punto:
    def __init__(self, x, y, cuadro_referencia):
        self.x = x
        self.y = y
        self.cuadro_referencia = cuadro_referencia

    def dibujar(self, screen):
        pygame.draw.circle(screen, BLUE, (self.cuadro_referencia.x + self.x, self.cuadro_referencia.y + self.y), 5)

# Crear la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Cuadro de Referencia con Puntos en Pygame")

# Crear el cuadro de referencia
cuadro_referencia = CuadroReferencia(100, 100, 400, 400)

# Crear una serie de puntos en relación con el cuadro de referencia
puntos = [Punto(50, 50, cuadro_referencia), Punto(100, 100, cuadro_referencia), Punto(150, 150, cuadro_referencia)]

# Bucle principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Limpiar la pantalla
    screen.fill(WHITE)

    # Dibujar el cuadro de referencia
    cuadro_referencia.dibujar(screen)

    # Dibujar los puntos en relación con el cuadro de referencia
    for punto in puntos:
        punto.dibujar(screen)

    pygame.display.flip()

# Salir del juego
pygame.quit()
sys.exit()
