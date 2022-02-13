import numpy as np


def phase_decode(binary_array):
    i = 0
    res = []
    sub = []
    leng = 1
    count = 0
    while i < len(binary_array):
        if count < leng:
            sub.append(binary_array[i])
            i += 1
            count += 1
        else:
            res.append(sub)
            leng += 1
            count = 1
            sub = []
            sub.append(binary_array[i])
            i += 1
    res.append(sub)
    return res


def decode(binary_array, num_phase):
    res = []
    num_gen = int(len(binary_array) // num_phase)
    for phase in range(num_phase):
        res.append(phase_decode(binary_array[phase * num_gen:phase * num_gen + num_gen]))
    return res


if __name__ == "__main__":
    n_phases = 1
    bit_string = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1])
    # print(bit_string)
    genome = decode(bit_string, 2)
    print(genome)
