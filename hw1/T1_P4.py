#####################
# CS 181, Spring 2021
# Homework 1, Problem 4
# Start Code
##################

import csv
import numpy as np
import matplotlib.pyplot as plt
import math

csv_filename = 'data/year-sunspots-republicans.csv'
years  = []
republican_counts = []
sunspot_counts = []

with open(csv_filename, 'r') as csv_fh:

    # Parse as a CSV file.
    reader = csv.reader(csv_fh)

    # Skip the header line.
    next(reader, None)

    # Loop over the file.
    for row in reader:

        # Store the data.
        years.append(float(row[0]))
        sunspot_counts.append(float(row[1]))
        republican_coun ts.append(float(row[2]))

# Turn the data into numpy arrays.
years  = np.array(years)
republican_counts = np.array(republican_counts)
sunspot_counts = np.array(sunspot_counts)
last_year = 1985

# Plot the data.
plt.figure(1)
plt.plot(years, republican_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")
plt.figure(2)
plt.plot(years, sunspot_counts, 'o')
plt.xlabel("Year")
plt.ylabel("Number of Sunspots")
plt.figure(3)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
# plt.show()

# Create the simplest basis, with just the time and an offset.
X = np.vstack((np.ones(years.shape), years)).T

# TODO: basis functions
# Based on the letter input for part ('a','b','c','d'), output numpy arrays for the bases.
# The shape of arrays you return should be: (a) 24x6, (b) 24x12, (c) 24x6, (c) 24x26
# xx is the input of years (or any variable you want to turn into the appropriate basis).
def make_basis(years, part, sunspot):
    x = 24
    y = 6
    if sunspot == True:
        x = 13
    if part == 'b':
        y = 12
    if part == 'd':
        y = 26
    output = np.empty([x, y])
    for i in range(len(years)):
        if part == 'a':
            row = np.empty([1, 6])
            row[0][0] = 1
            for j in range(6):
                if j != 0:
                    row[0][j] = years[i]**j
            output[i] = row
        if part == 'b':
            mu = 1960
            row = np.empty([1, 12])
            row[0][0] = 1
            for j in range(12):
                if j != 0:
                    row[0][j] = math.exp(-((years[i] - (mu + 5*j))**2)/25)
            output[i] = row
        if part == 'c':
            row = np.empty([1, 6])
            row[0][0] = 1
            for j in range(6):
                if j != 0:
                    row[0][j] = math.cos(years[i]/j)
            output[i] = row
        if part == 'd':
            row = np.empty([1, 26])
            row[0][0] = 1
            for j in range(26):
                if j != 0:
                    row[0][j] = math.cos(years[i]/j)
            output[i] = row
    print("THis is output \n")
    print(output)
    return output

# Nothing fancy for outputs.
Y = republican_counts

# Find the regression weights using the Moore-Penrose pseudoinverse.
def find_weights(X,Y):
    w = np.linalg.solve(np.dot(X.T, X) , np.dot(X.T, Y))
    print("These are the weights", w, np.shape(w))
    return w

def residual_square_error(y, y_hat, size):
    output = 0
    for i in range(size):
        output += (y[i] - y_hat[i])**2
    print("This is the res squared error", output)
    return output

# Compute the regression line on a grid of inputs.
# DO NOT CHANGE grid_years!!!!!
grid_years = np.linspace(1960, 2005, 200)
grid_X = np.vstack((np.ones(grid_years.shape), grid_years))
grid_Yhat  = np.dot(grid_X.T, find_weights(X, republican_counts))

grid_years_2 = np.linspace(1960, 2005, 24)
grid_sunspots = np.linspace(10, 160, 13)

grid_basis_Yhat_a = np.dot(make_basis(years, 'a', False), find_weights(make_basis(years, 'a', False), republican_counts))
grid_basis_Yhat_b = np.dot(make_basis(years, 'b', False), find_weights(make_basis(years, 'b', False), republican_counts))
grid_basis_Yhat_c = np.dot(make_basis(years, 'c', False), find_weights(make_basis(years, 'c', False), republican_counts))
grid_basis_Yhat_d = np.dot(make_basis(years, 'd', False), find_weights(make_basis(years, 'd', False), republican_counts))

residual_square_error(republican_counts, grid_basis_Yhat_a, 24)
residual_square_error(republican_counts, grid_basis_Yhat_b, 24)
residual_square_error(republican_counts, grid_basis_Yhat_c, 24)
residual_square_error(republican_counts, grid_basis_Yhat_d, 24)

# TODO: plot and report sum of squared error for each basis
plt.figure(4)
plt.plot(years, republican_counts, 'o', grid_years_2, grid_basis_Yhat_a, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")

plt.figure(5)
plt.plot(years, republican_counts, 'o', grid_years_2, grid_basis_Yhat_b, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")

plt.figure(6)
plt.plot(years, republican_counts, 'o', grid_years_2, grid_basis_Yhat_c, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")

plt.figure(7)
plt.plot(years, republican_counts, 'o', grid_years_2, grid_basis_Yhat_d, '-')
plt.xlabel("Year")
plt.ylabel("Number of Republicans in Congress")

# Plot the data and the regression line.
# plt.plot(years, republican_counts, 'o', grid_years, grid_Yhat, '-')
# plt.xlabel("Year")
# plt.ylabel("Number of Republicans in Congress")
# plt.show()

grid_basis_Yhat_sunspot_a = np.dot(make_basis(sunspot_counts[years<last_year], 'a', True), find_weights(make_basis(sunspot_counts[years<last_year], 'a', True), republican_counts[years<last_year]))
grid_basis_Yhat_sunspot_c = np.dot(make_basis(sunspot_counts[years<last_year], 'c', True), find_weights(make_basis(sunspot_counts[years<last_year], 'c', True), republican_counts[years<last_year]))
grid_basis_Yhat_sunspot_d = np.dot(make_basis(sunspot_counts[years<last_year], 'd', True), find_weights(make_basis(sunspot_counts[years<last_year], 'd', True), republican_counts[years<last_year]))

plt.figure(8)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_basis_Yhat_sunspot_a, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.figure(9)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_basis_Yhat_sunspot_c, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.figure(10)
plt.plot(sunspot_counts[years<last_year], republican_counts[years<last_year], 'o', grid_sunspots, grid_basis_Yhat_sunspot_d, '-')
plt.xlabel("Number of Sunspots")
plt.ylabel("Number of Republicans in Congress")
plt.show()

residual_square_error(republican_counts[years<last_year], grid_basis_Yhat_sunspot_a, 13)
residual_square_error(republican_counts[years<last_year], grid_basis_Yhat_sunspot_c, 13)
residual_square_error(republican_counts[years<last_year], grid_basis_Yhat_sunspot_d, 13)