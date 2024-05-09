from data_generation.sim_v2 import step
from data_generation.sim_v2_draw import draw

# just for fun, lets you fly a rocket in the simulation

import pygame
import numpy as np


def draw_rocket(throttle, surface, state, rocket_rot, color, size = 10, dist_scale = 1):
	direction = np.array([np.cos(rocket_rot), np.sin(rocket_rot)])
	ccw = np.array([-direction[1], direction[0]])
	pos = state[0, 0] * dist_scale + np.array([surface.get_width(), surface.get_height()]) / 2

	corners_relative = [
		direction * 2, 
		-direction + ccw,
		-direction - ccw
	]

	pygame.draw.polygon(surface, color, points = [corner * size + pos for corner in corners_relative])

	if throttle > 0:
		flame_corners_relative = [
			-direction + ccw / 2, 
			-direction - ccw / 2, 
			-direction * 1.5 * throttle - ccw / 2, 
			-direction * 1.5 * throttle + ccw / 2
		]
		pygame.draw.polygon(surface, [255, 80, 0], points=[corner * size + pos for corner in flame_corners_relative])


if __name__ == "__main__":

	state = np.empty((11, 2, 2))
	masses = np.empty(11)

	state[1:] = np.load("data/sol_state_vectors.npy")[:, :, :-1] # slice off the vertical axis
	masses[1:] = np.load("data/sol_masses.npy")

	pygame.init()

	angle = 0
	throttle = 0
	sim_speed = 0
	zoom = -10

	lock = 0

	colors = np.array(
		[	
			[255, 255, 255], 
			[47, 106, 105], 
			[100, 100, 100], 
			[255, 128, 0], 
			[26, 26, 26], 
			[230, 230, 230], 
			[153, 61, 0], 
			[176, 127, 53], 
			[176, 143, 54], 
			[85, 128, 170], 
			[54, 104, 150], 
		]
	)

	state[0] = np.array([
			[-2.869685126120390 * 10 ** 7, 1.450261006449537 * 10 ** 8], 
			[-2.246421200995819 * 10 ** 1, -3.815004596559553 * 10 ** 0]
		])

	masses[0] = 1

	# sizes = np.array([1, 5, 3, 15, 2, 4, 4, 8, 7, 6, 5])
	raw_sizes = np.array([
			0.0, 
			6378.137, 
			1738.0, 
			696000.0, 
			2440.0, 
			6051.84,
			3389.92, 
			71492.0, 
			60268.0, 
			25559.0, 
			24766.0
		])

	raw_sizes /= 2

	screen = pygame.display.set_mode((1280, 720))
	clock = pygame.time.Clock()
	running = True

	while running:

		sizes = raw_sizes * 10 ** zoom
		sizes[(sizes < 10) * (sizes > 0)] = 10

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		keys = pygame.key.get_pressed()

		if keys[pygame.K_z]:
			throttle = 1
		elif keys[pygame.K_x]:
			throttle = 0
		else:
			throttle += 0.05 * (keys[pygame.K_LSHIFT] - keys[pygame.K_LCTRL])
			throttle = max(0, min(1, throttle))

		angle += 0.05 * (keys[pygame.K_d] - keys[pygame.K_a])

		zoom += 0.05 * (keys[pygame.K_h] - keys[pygame.K_n])

		sim_speed += 0.05 * (keys[pygame.K_PERIOD] - keys[pygame.K_COMMA])
		sim_speed = max(0, sim_speed)

		if keys[pygame.K_SLASH]:
			sim_speed = 0

		step(state, masses = masses, dt = (10 ** sim_speed) / 60, lock = lock)

		state[0, 1] += np.array([np.cos(angle), np.sin(angle)]) * throttle * (10 ** sim_speed) / 60 * 0.05 / masses[0]

		screen.fill((255, 255, 255))
		draw_rocket(throttle, screen, state, angle, [0, 0, 0], dist_scale = 10 ** zoom)
		draw(state, screen, sizes = sizes, masses = masses, colors = colors, grav_scale = 2000, vel_scale = 0, dist_scale = 10 ** zoom)

		state[:] -= state[1]

		pygame.display.flip()

		clock.tick(60)

	pygame.quit()