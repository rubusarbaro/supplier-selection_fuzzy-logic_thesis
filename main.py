##################################################
##                                              ##
##      Saúl R. Morales © 2025 MIT License      ##
##                                              ##
##################################################


# Library import
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzzy


# Parameters
min_price = 0
max_price = 0
max_delivery = 60
max_response = 72


# Variables
price = np.arange(min_price,max_price + 1,1)
punctuality = np.arange(0,2,0.1)
delivery_time = np.arange(0,max_delivery + 1,1)
response_time = np.arange(0,max_response + 1,1)


# Membership functions
price_low = None
price_medium = None
price_high = None

punctuality_low = fuzzy.trapmf(punctuality, [0,0,0.25,0.5])
punctuality_medium = fuzzy.trimf(punctuality, [0.25,0.5,0.75])
punctuality_high = fuzzy.trapmf(punctuality, [0.5,0.75,1,1])

delivery_time_low = fuzzy.trapmf(delivery_time,[0,0,15,30])
delivery_time_medium = fuzzy.trimf(delivery_time, [15,30,45])
delivery_time_high = fuzzy.trapmf(delivery_time,[30,45,60,60])

response_time_low  = fuzzy.trapmf(response_time, [0, 0, 12, 36])
response_time_medium  = fuzzy.trimf(response_time, [24, 36, 48])
response_time_high = fuzzy.trapmf(response_time, [36, 48, 72, 72])


# Membership functions visualization
fig, (ax1,ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 9))

ax1.plot(punctuality, punctuality_low, "b", linewidth=1.5, label="Bad")
ax1.plot(punctuality, punctuality_medium, "g", linewidth=1.5, label="Regular")
ax1.plot(punctuality, punctuality_high, "r", linewidth=1.5, label="Good")
ax1.legend()

ax2.plot(delivery_time, delivery_time_low, "b", linewidth=1.5, label="Good")
ax2.plot(delivery_time, delivery_time_medium, "g", linewidth=1.5, label="Regular")
ax2.plot(delivery_time, delivery_time_high, "r", linewidth=1.5, label="Bad")
ax2.legend()

ax3.plot(response_time, response_time_low, "b", linewidth=1.5, label="Good")
ax3.plot(response_time, response_time_medium, "g", linewidth=1.5, label="Regular")
ax3.plot(response_time, response_time_high, "r", linewidth=1.5, label="Bad")
ax3.legend()

for ax in [ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()