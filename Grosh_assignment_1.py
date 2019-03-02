# Colin Grosh Assignment #1 DSCI 401
import math
from functools import *
import numpy as np

# Flatten Function
def flatten(lst):
    # Empty List to append the flattened list to
    output = []
    # Loop through the inputted list
    for i in lst:
        # Check if the value in the list is a list
        if type(i) == list:
            # If it is a list, put that list back into the flatten function and extend
            # the output list with the result of the flatten function
            output.extend(flatten(i))
        else:
            # If the value is not a list, append the value to the output list
            output.append(i)
    # return the output list
    return output

# Power Set Function
def power_set(lst):
    # Check if the length of the list is 0
    if len(lst) == 0:
        return [lst]
    # Check if the length of the list is 1
    if len(lst) == 1:
        return [lst]
    # If the length of the list is larger than 1, return the first index of the list
    # added to the rest of the list put back into the power set function, plus all
    # the rest of the list values after the first one
    return [[lst[0]] + s for s in power_set(lst[1:])] + [[x] for x in lst[1:]] + [[]]

# All Permentations Function
def all_perms(lst):
    # CHeck if the lenghth of the list is equal to 0
    if len(lst) == 0:
        return [[]]
    # Check if the length of the list is equal to 1
    if len(lst) == 1:
        return [lst]
    # If the length of the inputted list is greater than 1, declare an output list that
    # All the permentations will be added to
    output = []
    # Loop through the inputted list by index
    for i in range(len(lst)):
        # Segment the list to take the index of the current value,
        #and each of the other values in the list
        needed = lst[:i] + lst[1+i:]
        # Add the current list index to the front of each of the other values of the
        # list when inputted back into the permenations function
        q = [[lst[i]] + y for y in all_perms(needed)]
        # Add the list of permentations developed to the output list
        output.extend(q)
    # Return the output list with the permentations in it
    return output

# Spiral Function
# Me and Andrew worked somewhat together on this function and tried to get it to work
# Because neither of us were able to get it to work, but it was unsuccessfull,
# And this was the best attempt of what we had brainstormed
def spiral(n, corner):

    # Define the directions depending on each corner
    one_dir = [(0,-1), (1,0), (0,1), (-1,0), ()]
    two_dir = [(-1,0), (0,-1), (1,0), (0,1), ()]
    three_dir = [(1,0), (0,1), (-1,0), (0,-1), ()]
    four_dir = [(0,1), (0,-1), (0,-1), (1,0), ()]

    # Choose which direction set to use depending on what corner is entered
    if corner == 1:
        dir = one_dir.copy()
    elif corner == 2:
        dir = two_dir.copy()
    elif corner == 3:
        dir = three_dir.copy()
    elif corner == 4:
        dir = four_dir.copy()
    else:
        return "Corner must be within 1 and 4"

    use = n
    q = n**2
    changes = []
    count = n
    start = q-n
    other = (n**2)-1
    changes.append(start)

    # Get all the values in the matrix where the direction will change
    while count != 1:
        x = start - (n-1)
        y = x - (n-1)
        changes.append(x)
        changes.append(y)
        count -= 1
        n -= 1
        start = y

    value = []
    directions = []
    dir_index = 0

    # Get the values and the direction of each value depending on where is located
    while other > 0:
        if other in changes:
            value.append(other)
            directions.append(dir[dir_index])
            dir_index +=1
            if dir_index > 3:
                dir_index = 0
            other -= 1
        else:
            value.append(other)
            directions.append(dir[dir_index])
            other -= 1

    # Build the matrix
    matrix = [[0] * use for i in range(use)]

    # Determine where the first value will start based on what the corner was
    n = use
    if corner == 1:
        n_x = 0
        n_y = 0
    elif corner == 2:
        n_x = 0
        n_y = n-1
    elif corner == 3:
        n_x = n-1
        n_y = 0
    elif corner == 4:
        n_x = n-1
        n_y = n-1

    # Loop through and try to append the values to the correct position and
    # Add on to the index
    for i in range(0, (n**2)-2):
        print('it ran ', value[i])
        matrix[n_x][n_y] = value[i]
        if value[i] == 0:
            break
        else:
            n_x += directions[i+1][0]
            n_y += directions[i+1][1]
    for i in matrix:
        print(i)
