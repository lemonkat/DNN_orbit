import requests
import numpy as np
import datetime

# loads data from Horizons
# no need to run this


def datetime_to_string(time):
    return (
        f"{time.year}-{time.month}-{time.day}-{time.hour}:{time.minute}:{time.second}"
    )


BASE_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"



def fill_request_url(**kwargs):
    url = BASE_URL
    if len(kwargs) != 0:
        url += "?"
        arg_list = []
        for key, value in kwargs.items():
            arg_list.append(key + "=" + value)
        url += "&".join(arg_list)
    return url


def get_response_raw_text(id_code, t0, t1):
    url = fill_request_url(
        format="text",
        EPHEM_TYPE="VECTORS",
        OBJ_DATA="NO",
        COMMAND=f"'{id_code}'",
        VEC_TABLE="2",
        CAL_TYPE="GREGORIAN",
        CSV_FORMAT="YES",
        VEC_LABELS="NO",
        START_TIME=str(t0),
        STOP_TIME = str(t1),
        STEP_SIZE="1",
        CENTER="500@0",  # uncomment line for sol barycenter frame of reference, otherwise earth
    )
    return requests.get(url).text.replace("\t", " ")


def get_state_vector(text):
    line = text[text.index("$$SOE") + 6 :]
    line = line[: line.index("\n")]

    result = np.empty(6)

    nums = line.split(",")[2:]
    for i in range(6):
        result[i] = float(nums[i].replace(" ", ""))

    return result


def request_state_vector(id_code, time = datetime.datetime(2023, 1, 1, 12)):
    text = get_response_raw_text(
        id_code, 
        datetime_to_string(time), 
        datetime_to_string(time + datetime.timedelta(seconds=1))
    )
    flat_vector = get_state_vector(text)
    return np.array([flat_vector[:3], flat_vector[3:]])


if __name__ == "__main__":
    # order: earth, moon, sun, then all other planets in order
    selected_planets = [399, 301, 10, 199, 299, 499, 599, 699, 799, 899]

    num = len(selected_planets)
    SOL_MASSES = np.array(
        [
            5.97219 * (10**24),  # earth
            7.349 * (10**22),  # moon
            1988500.0 * (10**24),  # sun
            3.302 * (10**23),  # mercury
            48.685 * (10**23),  # venus
            6.4171 * (10**23),  # mars
            189818.722 * (10**22),  # jupiter
            5.6834 * (10**26),  # saturn
            86.813 * (10**24),  # uranus
            102.409 * (10**24),  # neptune
            # 1.307 * (10 ** 22) # pluto
        ]
    )
    SOL_STATE_VECTORS = np.empty((num, 2, 3))

    for i, id_num in enumerate(selected_planets):
        SOL_STATE_VECTORS[i] = request_state_vector(id_num)

    np.save("data/sol_masses.npy", SOL_MASSES)
    np.save("data/sol_state_vectors.npy", SOL_STATE_VECTORS)

    # print(request_state_vector("-138250"))
