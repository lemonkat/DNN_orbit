import numpy as np
from scipy.interpolate import CubicSpline
from math import ceil

# cubic spline interpolator thing

class ContinousInterpolator:
    def __init__(self, start_state, model, t_step):
        self.model = model
        self.t_step = t_step
        self.data = np.array([start_state], dtype=np.float32)
        self.func = None

        self.shape = start_state.shape

        self.ready(self.t_step * 10)
        
    def ready(self, max_time):
        new_data = np.empty((ceil(max_time / self.t_step), *self.shape), dtype=np.float32)
        new_data[:len(self.data)] = self.data
        
        for i in range(len(self.data), len(new_data)):
            new_data[i] = self.model.predict(new_data[i - 1])

        self.data = new_data

        self.func = CubicSpline(
            np.arange(0, max_time, self.t_step), 
            self.data
        )

    def __call__(self, t):
        if self.func is None or t > self.data.shape[0] * self.t_step:
            self.ready(t)

        return self.func(t)

if __name__ == "__main__":
    import pygame
    from load_utils import load_model
    from data_generation.orbit_generator import generate_state_vectors 
    from data_generation.partial import combine_solutions

    generate_new_data = False

    if generate_new_data:
        state = generate_state_vectors(100, dims=3, flat=True)
        np.save("data/A_S_UpdateTemp.npy", state)
    else:
        state = np.load("data/A_S_UpdateTemp.npy")

    model = load_model(f"saved_models/ANN_06_60B")
    solution = ContinousInterpolator(state, model, 60)

    pygame.init()

    screen_size = np.array((1280, 720))

    screen = pygame.display.set_mode(screen_size)
    clock = pygame.time.Clock()

    running = True

    def convert(point):
        return (point[0, :-1] / 50).astype(int) + screen_size / 2 

    t = 0
    points = []

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("white")



        points.append(convert(solution(t)))

        place_image(screen, screen_size / 2, "data/earth.png", (40, 40))

        pg.draw.lines(
            screen, 
            [0, 0, 0], 
            False, 
            points
        )

        place_image(screen, points[-1], "data/sat_icon.png", (25, 25))


        # text = font.render(f"T = {t} S", False, [0, 0, 0])
        # screen.blit(text, (10, 10))

        # text = font.render(f"ALT = {round_3digit(np.sqrt(np.sum(points[-1][0] ** 2)) - 5000)} KM", False, [0, 0, 0])
        # screen.blit(text, (10, 30))

        # text = font.render(f"SPD = {round_3digit(np.sqrt(np.sum(points[-1][1] ** 2)))} KM/S", False, [0, 0, 0])
        # screen.blit(text, (10, 50))
        

        # flip() the display to put your work on screen
        pygame.display.flip()

        clock.tick(60)  # limits FPS to 60

        # if len(points) > 1000:
        #     points = points[-1000:]

        t += 10

        

    pygame.quit()




