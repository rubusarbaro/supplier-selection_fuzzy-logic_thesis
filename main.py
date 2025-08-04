##################################################
##                                              ##
##      Saúl R. Morales © 2025 MIT License      ##
##                                              ##
##################################################

# Library import
import numpy as np
import skfuzzy as fuzzy

# Parameters
max_price = 0
max_delivery = 0
max_response = 0

# Variables
price = np.arange(0,max_price + 1,1)
punctuality = np.arange(0,2,0.1)
delivery_time = np.arange(0,max_delivery,1)
response_time = np.arange(0,max_response,1)