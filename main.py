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
price_µ = 0
price_σ = 0
max_price = 0
punctuality_µ = 0
punctuality_σ = 0
delivery_time_µ = 0
delivery_time_σ = 0
max_delivery_time = 60
quotation_time_µ = 0
quotation_time_σ = 0
max_quotation_time = 30


# Variables
price = np.arange(0, max_price + 1, 0.01)
punctuality = np.arange(0, 2, 0.1)
delivery_time = np.arange(0, max_delivery_time + 1, 1)
quotation_time = np.arange(0, max_quotation_time + 1, 1)


# Membership functions
price_low = fuzzy.trapmf(price, [0, 0, price_μ - price_σ, price_μ])
price_medium = fuzzy.trimf(price, [price_μ - price_σ, price_μ, price_μ + price_σ])
price_high = fuzzy.trapmf(price, [price_μ, price_μ + price_σ,max_price, max_price])

punctuality_low = fuzzy.trapmf(punctuality, [0,0,0.25,0.5])
punctuality_medium = fuzzy.trimf(punctuality, [0.25,0.5,0.75])
punctuality_high = fuzzy.trapmf(punctuality, [0.5,0.75,1,1])

delivery_time_low = fuzzy.trapmf(delivery_time,[0,0,15,30])
delivery_time_medium = fuzzy.trimf(delivery_time, [15,30,45])
delivery_time_high = fuzzy.trapmf(delivery_time,[30,45,60,60])

quotation_time_low  = fuzzy.trapmf(quotation_time, [0, 0, 12, 36])
quotation_time_medium  = fuzzy.trimf(quotation_time, [24, 36, 48])
quotation_time_high = fuzzy.trapmf(quotation_time, [36, 48, 72, 72])


# Membership functions visualization
fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(price, price_low, "b", linewidth=1.5, label="Low")
ax0.plot(price, price_medium, "g", linewidth=1.5, label="Regular")
ax0.plot(price, price_high, "r", linewidth=1.5, label="High")
ax0.legend()

ax1.plot(punctuality, punctuality_low, "b", linewidth=1.5, label="Bad")
ax1.plot(punctuality, punctuality_medium, "g", linewidth=1.5, label="Regular")
ax1.plot(punctuality, punctuality_high, "r", linewidth=1.5, label="Good")
ax1.legend()

ax2.plot(delivery_time, delivery_time_low, "b", linewidth=1.5, label="Good")
ax2.plot(delivery_time, delivery_time_medium, "g", linewidth=1.5, label="Regular")
ax2.plot(delivery_time, delivery_time_high, "r", linewidth=1.5, label="Bad")
ax2.legend()

ax3.plot(quotation_time, quotation_time_low, "b", linewidth=1.5, label="Good")
ax3.plot(quotation_time, quotation_time_medium, "g", linewidth=1.5, label="Regular")
ax3.plot(quotation_time, quotation_time_high, "r", linewidth=1.5, label="Bad")
ax3.legend()

for ax in [ax0, ax1, ax2, ax3]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()