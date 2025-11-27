import pandas as pd
import numpy as np
from datetime import date, timedelta

# Load the data from the uploaded CSV file
df = pd.read_csv('Nat_Gas.csv')

# Convert the 'Dates' column to datetime objects
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Sort the data by date to ensure it's in chronological order
df.sort_values('Dates', inplace=True)
df.reset_index(drop=True, inplace=True)

# Extract the price data and dates
prices = df['Prices'].values
dates = df['Dates'].values

# Define the start and end dates for our analysis
start_date = date(2020, 10, 31)
end_date = date(2024, 9, 30)

# Create a list of all the month-end dates in our dataset
months = []
year = start_date.year
month = start_date.month + 1
while True:
    # Get the last day of the current month
    current = date(year, month, 1) + timedelta(days=-1)
    months.append(current)
    if current.month == end_date.month and current.year == end_date.year:
        break
    else:
        month = ((month + 1) % 12) or 12
        if month == 1:
            year += 1

# Calculate the number of days from the start date for each data point
days_from_start = [(day - start_date).days for day in months]
time = np.array(days_from_start)

# Function for simple linear regression (y = Ax + B)
def simple_regression(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept

# Fit the linear trend to the data
slope, intercept = simple_regression(time, prices)

# Calculate the residuals (the part of the price not explained by the linear trend)
sin_prices = prices - (time * slope + intercept)

# Define the seasonal component using a sine wave with a 1-year (365-day) period
sin_time = np.sin(time * 2 * np.pi / 365)
cos_time = np.cos(time * 2 * np.pi / 365)

# Function for bilinear regression (to fit the sine and cosine components)
def bilinear_regression(y, x1, x2):
    # Projection onto the x-vectors
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

# Fit the seasonal component
slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)

# Calculate the amplitude and phase shift for the sine wave
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift = np.arctan2(slope2, slope1)

# Define the final interpolation/extrapolation function
def estimate_price(date_str):
    """
    Estimate the natural gas price for a given date.
    The date should be a string in 'YYYY-MM-DD' format.
    Returns the estimated price as a float.
    """
    target_date = pd.to_datetime(date_str, format='%Y-%m-%d')
    days = (target_date - pd.Timestamp(start_date)).days
    
    # Calculate the price using the fitted model: linear trend + seasonal component
    estimated_price = amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept
    return estimated_price

# Example usage:
# print(estimate_price("2025-06-15"))  # Extrapolate into the future
# print(estimate_price("2022-07-15"))  # Interpolate within the data range