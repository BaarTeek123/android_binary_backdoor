import numpy as np
from random import random
# add a random number

def add_random_binary(row, random_value):
    binary_str = ''.join(str(x) for x in row)
    decimal = int(binary_str, 2)
    new_decimal = decimal + random_value
    new_binary_str = bin(new_decimal)[2:]  # remove '0b' prefix
    new_binary = [int(x) for x in new_binary_str]
    return new_binary[:len(binary_str)]

# add a noise provided by an argument

def create_random_trigger(row_length, trigger_length, immutable_possitions):
	modification_order = np.argsort(uniform_noise(row_length))
	set_on = signal_length * '0'
 	set_off = signal_length * '1'
	for possition in modification_order:
		if not trigger_length:
			break
		if possition in immutable_possitions:
			continue
		trigger_length -= 1
		if random() < 0.5:
			set_off[possition] = '0'
		else:
			set_on[possition] = '1'
	return (int(set_on, 2), int(set_off, 2))		

def apply_trigger(row, trigger):
	set_on, set_off = trigger
	binary_str = ''.join(str(x) for x in row)
    decimal = int(binary_str, 2)
    new_decimal = set_on | decimal
    new_decimal ^= set_off
    new_binary_str = bin(new_decimal)[2:]  # remove '0b' prefix
    new_binary = [int(x) for x in new_binary_str]
    return new_binary[:len(binary_str)]
    
 
def add_noise(row, noise_function):
    noise = noise_function(len(row))
    row_with_noise = np.clip(row + noise, 0, 1)  
    return np.round(row_with_noise).astype(int)  

def gaussian_noise(length, std_dev=0.1):
    return np.random.normal(0, std_dev, length)

def uniform_noise(length, range=0.1):
    return np.random.uniform(-range, range, length)
