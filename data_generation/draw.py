from data_generation.sim import get_accel_vectors
from data_generation.advanced import simulate_RK45
from data_generation.partial import combine_solutions, simulate_RK45_partial
from data_generation.orbit_generator import generate_state_vectors as generate
import numpy as np
import pygame
import pickle

# displays things
# sizes is display size
# don't use

sat_icon = pygame.transform.scale(pygame.image.load("data/sat_icon.png"), (50, 50))
pygame.font.init()
font = pygame.font.Font(pygame.font.get_default_font(), 15)


def draw_arrow(surface, pos, vect, color, width=1):
    if np.sum(np.abs(vect)) == 0:
        return
    end = pos + vect
    direction = vect / np.sqrt(np.sum(vect**2))

    pygame.draw.line(surface, color, pos, end, width=width)

    offset = 10 * np.array([-direction[1], direction[0]])

    points = [end, end - (direction * 10) + offset, end - (direction * 10) - offset]

    pygame.draw.polygon(surface, color, points=points)


def draw_sat(surface, pos, sat_id=None):
    surface.blit(sat_icon, pos - np.array([25, 25]))

    # if sat_id is not None:
    #     text = font.render(str(sat_id), False, [0, 0, 0])
    #     surface.blit(text, pos - np.array(font.size(str(sat_id))) / 2)

def place_image(screen, pos, filename, size):
    img = pygame.transform.scale(pygame.image.load(filename), np.array(size) * 2)
    screen.blit(img, pos -  np.array(size))


def draw(
    state,
    surface,
    sizes=None,
    masses=None,
    grav_scale=0,
    vel_scale=0,
    colors=None,
    arrow_scale=10,
    dist_scale=10,
    lock=None,
    min_size=20,
):
    num_planets = len(sizes)

    sat_num = 0

    center = np.array([surface.get_width(), surface.get_height()]) / 2

    if state.shape[2] > 2:
        state = state[..., :2]

    if colors is None:
        colors = generate_colors(num_planets)

    for i in range(num_planets):
        pygame.draw.circle(
            surface, colors[i], state[i, 0] * dist_scale + center, max(min_size, sizes[i] * dist_scale)
        )

    for sat_idx in range(num_planets, len(state)):
        draw_sat(surface, state[sat_idx, 0] * dist_scale + center, sat_id = sat_idx - num_planets)

    if grav_scale > 0:
        accel_vectors = get_accel_vectors(state[:, 0])
        accel_vectors /= np.sqrt(np.sum(accel_vectors ** 2, axis=1))[..., None]
        for i in range(len(state)):
            draw_arrow(
                surface,
                state[i, 0] * dist_scale + center,
                accel_vectors[i] * grav_scale,
                [255, 0, 0],
            )

    if vel_scale > 0:
        for i in range(len(state)):
            draw_arrow(
                surface,
                state[i, 0] * dist_scale + center,
                state[i, 1] * vel_scale,
                [0, 0, 255],
            )
            
    place_image(surface, state[0, 0] * dist_scale + center, "data/earth.png", (4000 * 0.02, 4000 * 0.02))
    #place_image(surface, state[0, 0] * dist_scale + center, "data/earth.png", (30, 30))
    #place_image(surface, state[1, 0] * dist_scale + center, "data/moon.png", (20, 20))
    #place_image(surface, state[2, 0] * dist_scale + center, "data/sun.png", (50, 50))




# construct on the surface and iterate
# remember to clear the screen and flip the display
def draw_ode_solution(solution, surface, time, dt=1, lock=None, **kwargs):
    for t in range(0, time, dt):
        state = solution(t)
        if lock is not None:
            state -= state[lock]
        draw(state, surface, lock=lock, **kwargs)
        yield dt


if __name__ == "__main__":

    colors = np.array(
        [
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

    # if a size is 0, that means that it is a satellite
    sizes = np.array([
        6378.137, 
        1738.0, 
        696000.0, 
        2440.0, 
        6051.84,
        3389.92, 
        71492.0, 
        60268.0, 
        25559.0, 
        24766.0,
    ])

    sizes /= 2 # diameter->radius

    sizes *= 0.0001

    pygame.init()

    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()

    running = True

    time = 500000

    # select which line to uncomment to select the zoom level
    dist_scale = 0.02 # LEO
    # dist_scale = 0.0002 # earth-moon
    # dist_scale = 0.0000015 # inner planets
    # dist_scale = 0.0000001 # outer planets

    #planet_selection = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    planet_selection = np.array([0, 1, 2])

    masses = np.load("data/sol_masses.npy")[planet_selection]


    planet_solution = simulate_RK45(
        np.load("data/sol_state_vectors.npy")[planet_selection, ..., :],
        time,
        masses=masses,
        dt=1,
        lock=0
    )

    solution = planet_solution

    sat_solution = simulate_RK45_partial(
        np.load("data/A_S_UpdateTemp.npy")[:1],
        time,
        planet_solution,
        dt=1,
        masses=masses
    )
    

    solution = combine_solutions(planet_solution, sat_solution)
    

    iterator = draw_ode_solution(
        solution,
        screen,
        time,
        dt=10,
        dist_scale=dist_scale,
        masses=masses,
        sizes=sizes[planet_selection],
        colors=colors[planet_selection],
        #vel_scale=7,
        #grav_scale = 35, 
    )

    input("Simulations ready. Confirm? ")
    for output in iterator:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        # delta-time is 1 hour
        # 60 FPS, so speed is 216000x real-time

        pygame.display.flip()
        screen.fill((255, 255, 255))
        clock.tick(60)

    pygame.quit()
