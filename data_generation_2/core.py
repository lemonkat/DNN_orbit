import numpy as np

# km^3 / kg / s^2


# the core of the simulator
# things are measured in kilometers, kilograms, seconds

def get_accel_vectors(positions, masses=None):
    num_planets = len(positions)

    # masses defaults to all 1kg
    if masses is None:
        masses = np.ones(num_planets)

    each = np.arange(num_planets)

    # get distances all at once
    p1s, p2s = np.meshgrid(each, each, indexing="ij")

    # distance_vectors[i, j] = the vector from planet i to planet j
    distance_vectors = positions[p2s] - positions[p1s]

    # dist_squareds[i, j] = the square of the distance from planet i to planet j
    dist_squareds = np.sum(distance_vectors**2, axis=2)

    ignore = (
        dist_squareds == 0
    )  # things pulling on themselves break things, but they should be zero

    dist_squareds[ignore] = 1

    # calculate pull strengths
    # pulls[i, j] = the magnitude of the
    # free-fall acceleration of planet i caused by planet j
    pulls = masses[p2s] * GRAV_CONSTANT / dist_squareds

    # get pull vectors and sum them for total accel
    # pull_vectors[i, j] is the vector of the free-fall
    # acceleration of planet i caused by planet j
    pull_vectors = pulls[..., None] * (
        distance_vectors / np.sqrt(dist_squareds)[..., None]
    )

    pull_vectors[ignore] = 0

    # sum up pulls from each other planet
    # accel_vectors[i] = the combined free-fall acceleration of planet j
    accel_vectors = np.sum(pull_vectors, axis=1)

    return accel_vectors
