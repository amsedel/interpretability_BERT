import tkinter as tk
import pygame
from pygame.locals import *
from constants import *

# Función para iniciar pygame
def iniciar_pygame():
    pygame.init()
    pantalla = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Ejemplo de Pygame")
    
    corriendo = True
    while corriendo:
        for evento in pygame.event.get():
            if evento.type == QUIT:
                corriendo = False

        pantalla.fill((255, 255, 255))
        pygame.draw.circle(pantalla, (255, 0, 0), (200, 150), 50)
        pygame.display.update()

    pygame.quit()

# Función para iniciar pygame sin cerrar la ventana de tkinter
def iniciar_pygame_sin_cerrar_tkinter():
    ventana_pygame = tk.Toplevel(ventana_tkinter)
    iniciar_pygame()
    ventana_pygame.destroy()  # Cierra la ventana secundaria de pygame

# Crear ventana de tkinter
ventana_tkinter = tk.Tk()
ventana_tkinter.title("Interfaz de tkinter con Pygame")

# Crear botón en tkinter para iniciar pygame sin cerrar la ventana de tkinter
boton_iniciar_pygame = tk.Button(ventana_tkinter, text="Iniciar Pygame", command=iniciar_pygame_sin_cerrar_tkinter)
boton_iniciar_pygame.pack()

# Iniciar el bucle principal de tkinter
ventana_tkinter.mainloop()
