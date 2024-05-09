from data_generation.orbit_generator import generate_state_vectors
from load_utils import load_model
from noise_utils import add_noise

import tensorflow as tf
import numpy as np
import pygame
import sys
from tqdm import tqdm

tgt_time = int(sys.argv[1])

generate_new_data = False

model = load_model(f"saved_models/ANN_06_{tgt_time}_FINAL")
#model = load_model(f"saved_models/ANN_07_{tgt_time}C")

if generate_new_data:
    state = generate_state_vectors(100, dims=3, flat=True)
    np.save("data/A_S_UpdateTemp.npy", state)
else:
    state = np.load("data/A_S_UpdateTemp.npy")

state = state[:2]
state[1] = add_noise(np.array([state[0]]), 0.05, 0)[0]



screen_size = np.array((1280, 720), dtype=np.int16)



# pygame setup
pygame.init()
screen = pygame.display.set_mode(screen_size)
clock = pygame.time.Clock()
running = True

pygame.font.init()
font = pygame.font.Font(pygame.font.get_default_font(), 15)

def place_image(screen, pos, filename, size):
    img = pygame.transform.scale(pygame.image.load(filename), np.array(size) * 2)
    screen.blit(img, pos -  np.array(size))


points = [state]


def convert(point):
    return (point[0, :-1] / 50).astype(int) + screen_size / 2 


t = 0

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    state = model.predict(state, verbose=False)
    points.append(state)
    t += tgt_time

    screen.fill("white")


    place_image(screen, screen_size / 2, "data/earth.png", (40, 40))

    pygame.draw.lines(
        screen, 
        [0, 0, 255], 
        False, 
        [convert(s[0]) for s in points]
    )

    pygame.draw.lines(
        screen, 
        [255, 0, 0], 
        False, 
        [convert(s[1]) for s in points]
    )

    place_image(screen, convert(points[-1][0]), "data/sat_icon.png", (20, 20))
    place_image(screen, convert(points[-1][1]), "data/sat_icon_red.png", (20, 20))
    

    # flip() the display to put your work on screen
    pygame.display.flip()

    #clock.tick(60)  # limits FPS to 60

    # if len(points) > 1000:
    #     points = points[-1000:]

    

pygame.quit()