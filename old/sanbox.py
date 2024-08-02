data = {'left': {'x': [1, 2, 3], 'y': [4, 5, 6]}, 'right': {'x': [2, 3, 4], 'y': [5, 6, 7]}}

# Combine all the unique x and y values into a single list
all_values = list(set([*data['left']['x'], *data['left']['y'], *data['right']['x'], *data['right']['y']]))

print(all_values)