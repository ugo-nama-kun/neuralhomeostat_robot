import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Define the polygon vertices
polygon_vertices = [(200, 200), (400, 200), (400, 400), (200, 400)]
polygon_color = pygame.Color('blue')

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw the polygon
    pygame.draw.polygon(screen, polygon_color, polygon_vertices)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit the game
pygame.quit()
