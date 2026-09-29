import csv
import numpy as np
import math 


def get_data(csv_file_path):
    rows = []
    with open(csv_file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        
        for row in reader:
            rows.append(row)

    labels = [row[0] for row in rows]
    coords = [list(map(float, row[1:])) for row in rows]

    letters = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) not in ["J", "Z"]]
    letter_idx = {label: idx for idx, label in enumerate(letters)}


    inputs = np.array(coords, dtype=np.float32)
    outputs = np.zeros((len(labels), len(letters)), dtype=np.float32)


    for i, label in enumerate(labels):
        outputs[i, letter_idx[label]] = 1.0


    n = inputs.shape[0]

    indices = np.random.permutation(inputs.shape[0])

    inputs = inputs[indices]
    outputs = outputs[indices]


    training_size = math.floor(n * 0.8)


    training_inputs = inputs[:training_size, :]
    training_outputs = outputs[:training_size, :]

    testing_inputs = inputs[training_size:, :]
    testing_outputs = outputs[training_size:, :]


    return training_inputs, training_outputs, testing_inputs, testing_outputs