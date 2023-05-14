import numpy as np

# add a random number

def add_random_binary(row, random_value):
    binary_str = ''.join(str(x) for x in row)
    decimal = int(binary_str, 2)
    new_decimal = decimal + random_value
    new_binary_str = bin(new_decimal)[2:]  # remove '0b' prefix
    new_binary = [int(x) for x in new_binary_str]
    return new_binary[:len(binary_str)]

# add a noise provided by an argument
 
def add_noise(row, noise_function):
    noise = noise_function(len(row))
    row_with_noise = np.clip(row + noise, 0, 1)  
    return np.round(row_with_noise).astype(int)  

def gaussian_noise(length, std_dev=0.1):
    return np.random.normal(0, std_dev, length)

def uniform_noise(length, range=0.1):
    return np.random.uniform(-range, range, length)
