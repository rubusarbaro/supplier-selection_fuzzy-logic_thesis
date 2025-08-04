##################################################
##                                              ##
##      Saúl R. Morales © 2025 MIT License      ##
##                                              ##
##################################################


# Library import
import numpy as np
import skfuzzy as fuzzy


# Parameters
min_price = 0
max_price = 0
max_delivery = 0
max_response = 0


# Variables
price = np.arange(min_price,max_price + 1,1)
punctuality = np.arange(0,2,0.1)
delivery_time = np.arange(0,max_delivery,1)
response_time = np.arange(0,max_response,1)


# Membership functions
price_low = None
price_medium = None
price_high = None

punctuality_low = fuzzy.trimf(punctuality, [0, 0, 0.5])
punctuality_medium = fuzzy.trimf(punctuality, [0, 0.5, 1])
punctuality_high = fuzzy.trimf(punctuality, [0.5, 1, 1])

delivery_time_low = None
delivery_time_medium = None
delivery_time_high = None

response_time_low = None
response_time_medium = None
response_time_high = None