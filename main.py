##################################################
##                                              ##
##      Saúl R. Morales © 2025 MIT License      ##
##                                              ##
##################################################


# Library import
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzzy


# Inputs
price = 10.54
punctuality = 0.6
delivery_time = 7
quotation_time = 8


# Parameters
price_µ = 400
price_σ = 200
max_price = 1200
#punctuality_µ = 0
#punctuality_σ = 0
#delivery_time_µ = 0
#delivery_time_σ = 0
#max_delivery_time = 60
#quotation_time_µ = 0
#quotation_time_σ = 0
#max_quotation_time = 30


# Variables
var_price = np.arange(0, max_price + 1, 0.01)
var_punctuality = np.arange(0, 2, 0.1)
var_delivery_time = np.arange(0, 61, 1)
var_quotation_time = np.arange(0, 73, 1)
var_supplier = np.arange(0, 11, 0.01)


# Membership functions
price_low = fuzzy.trapmf(var_price, [0, 0, price_μ - price_σ, price_μ])
price_medium = fuzzy.trimf(var_price, [price_μ - price_σ, price_μ, price_μ + price_σ])
price_high = fuzzy.trapmf(var_price, [price_μ, price_μ + price_σ,max_price, max_price])

punctuality_low = fuzzy.trapmf(var_punctuality, [0,0,0.25,0.5])
punctuality_medium = fuzzy.trimf(var_punctuality, [0.25,0.5,0.75])
punctuality_high = fuzzy.trapmf(var_punctuality, [0.5,0.75,1,1])

delivery_time_low = fuzzy.trapmf(var_delivery_time,[0,0,15,30])
delivery_time_medium = fuzzy.trimf(var_delivery_time, [15,30,45])
delivery_time_high = fuzzy.trapmf(var_delivery_time,[30,45,60,60])

quotation_time_low  = fuzzy.trapmf(var_quotation_time, [0, 0, 12, 36])
quotation_time_medium  = fuzzy.trimf(var_quotation_time, [24, 36, 48])
quotation_time_high = fuzzy.trapmf(var_quotation_time, [36, 48, 72, 72])

supplier_low = fuzzy.trimf(var_supplier, [0, 2.5, 5])
supplier_medium = fuzzy.trimf(var_supplier, [2.5, 5, 7.5])
supplier_high = fuzzy.trimf(var_supplier, [5, 7.5, 10])


# Assign membership degree
price_level_low = fuzzy.interp_membership(var_price, price_low, price)
price_level_medium = fuzzy.interp_membership(var_price, price_medium, price)
price_level_high = fuzzy.interp_membership(var_price, price_high, price)

punctuality_level_low = fuzzy.interp_membership(var_punctuality, punctuality_low, punctuality)
punctuality_level_medium = fuzzy.interp_membership(var_punctuality, punctuality_medium, punctuality)
punctuality_level_high = fuzzy.interp_membership(var_punctuality, punctuality_high, punctuality)

delivery_time_level_low = fuzzy.interp_membership(var_delivery_time, delivery_time_low, delivery_time)
delivery_time_level_medium = fuzzy.interp_membership(var_delivery_time, delivery_time_medium, delivery_time)
delivery_time_level_high = fuzzy.interp_membership(var_delivery_time, delivery_time_high, delivery_time)

quotation_time_level_low = fuzzy.interp_membership(var_quotation_time, quotation_time_low, quotation_time)
quotation_time_level_medium = fuzzy.interp_membership(var_quotation_time, quotation_time_medium, quotation_time)
quotation_time_level_high = fuzzy.interp_membership(var_quotation_time, quotation_time_high, quotation_time)


# Rule application
# Example code uses np.fmax for OR operator. I will use np.fmin for AND.
rule_1 = np.fmin.reduce([price_level_low, punctuality_level_high, delivery_time_level_low, quotation_time_level_low])

rule_2 = np.fmin.reduce([price_level_medium, punctuality_level_high, delivery_time_level_low, quotation_time_level_low])

rule_3 = np.fmin.reduce([price_level_low, punctuality_level_medium, delivery_time_level_low, quotation_time_level_low])

rule_4 = np.fmin.reduce([price_level_low, punctuality_level_high, delivery_time_level_medium, quotation_time_level_low])

rule_5 = np.fmin.reduce([price_level_low, punctuality_level_high, delivery_time_level_low, quotation_time_level_medium])

rule_6 = np.fmin.reduce([price_level_high, punctuality_level_low, delivery_time_level_high, quotation_time_level_high])

rule_7 = np.fmin.reduce([price_level_medium, punctuality_level_low, delivery_time_level_high, quotation_time_level_high])

rule_8 = np.fmin.reduce([price_level_high, punctuality_level_medium, delivery_time_level_high, quotation_time_level_high])

rule_9 = np.fmin.reduce([price_level_high, punctuality_level_low, delivery_time_level_medium, quotation_time_level_high])

rule_10 = np.fmin.reduce([price_level_high, punctuality_level_low, delivery_time_level_high, quotation_time_level_medium])

rule_11 = 1 - np.fmax.reduce([rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10])

supplier_activation_low = np.fmin.reduce([
    np.fmin(rule_6, supplier_low),
    np.fmin(rule_7, supplier_low),
    np.fmin(rule_8, supplier_low),
    np.fmin(rule_9, supplier_low),
    np.fmin(rule_10, supplier_low)
    ])

supplier_activation_medium = np.fmin(rule_11, supplier_medium)

supplier_activation_high = np.fmin.reduce([
    np.fmin(rule_1, supplier_low),
    np.fmin(rule_2, supplier_low),
    np.fmin(rule_3, supplier_low),
    np.fmin(rule_4, supplier_low),
    np.fmin(rule_5, supplier_low)
    ])

supplier_0 = np.zeros_like(var_supplier)

aggregated = np.fmax.reduce([supplier_activation_low, supplier_activation_medium, supplier_activation_high])


# Defuzzification
supplier_score = fuzzy.defuzz(var_supplier, aggregated, "centroid")
supplier_activation = fuzzy.interp_membership(var_supplier, aggregated, supplier_score)


## Plots
if True:
    # One plot in one
    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=7, figsize=(8, 9))

    # Membership functions visualization
    ax0.plot(var_price, price_low, "b", linewidth=1.5, label="Low")
    ax0.plot(var_price, price_medium, "g", linewidth=1.5, label="Regular")
    ax0.plot(var_price, price_high, "r", linewidth=1.5, label="High")
    ax0.set_title("Price")
    ax0.legend()

    ax1.plot(var_punctuality, punctuality_low, "b", linewidth=1.5, label="Bad")
    ax1.plot(var_punctuality, punctuality_medium, "g", linewidth=1.5, label="Regular")
    ax1.plot(var_punctuality, punctuality_high, "r", linewidth=1.5, label="Good")
    ax1.set_title("Punctuality")
    ax1.legend()

    ax2.plot(var_delivery_time, delivery_time_low, "b", linewidth=1.5, label="Good")
    ax2.plot(var_delivery_time, delivery_time_medium, "g", linewidth=1.5, label="Regular")
    ax2.plot(var_delivery_time, delivery_time_high, "r", linewidth=1.5, label="Bad")
    ax2.set_title("Delivery time")
    ax2.legend()

    ax3.plot(var_quotation_time, quotation_time_low, "b", linewidth=1.5, label="Good")
    ax3.plot(var_quotation_time, quotation_time_medium, "g", linewidth=1.5, label="Regular")
    ax3.plot(var_quotation_time, quotation_time_high, "r", linewidth=1.5, label="Bad")
    ax3.set_title("Quotation time")
    ax3.legend()

    ax4.plot(var_supplier, supplier_low, "b", linewidth=1.5, label="Bad")
    ax4.plot(var_supplier, supplier_medium, "g", linewidth=1.5, label="Regular")
    ax4.plot(var_supplier, supplier_high, "r", linewidth=1.5, label="Good")
    ax4.set_title("Supplier")
    ax4.legend()

    # Membership and result
    ax5.fill_between(var_supplier, supplier_0, supplier_activation_low, facecolor="b", alpha=0.7)
    ax5.plot(var_supplier, supplier_low, "b", linewidth=0.5, linestyle="--")
    ax5.fill_between(var_supplier, supplier_0, supplier_activation_medium, facecolor="g", alpha=0.7)
    ax5.plot(var_supplier, supplier_medium, "g", linewidth=0.5, linestyle="--")
    ax5.fill_between(var_supplier, supplier_0, supplier_activation_high, facecolor="r", alpha=0.7)
    ax5.plot(var_supplier, supplier_high, "r", linewidth=0.5, linestyle="--")
    ax5.set_title("Output membership")

    ax6.plot(var_supplier, supplier_low, "b", linewidth=0.5, linestyle="--")
    ax6.plot(var_supplier, supplier_medium, "g", linewidth=0.5, linestyle="--")
    ax6.plot(var_supplier, supplier_high, "r", linewidth=0.5, linestyle="--")
    ax6.fill_between(var_supplier, supplier_0, aggregated, facecolor="Orange", alpha=0.7)
    ax6.plot([supplier_score, supplier_score], [0, supplier_activation], "k", linewidth=1.5, alpha=0.9)
    ax6.set_title("Aggregated membership and supplier score (line)")

    # Plot display
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()

print(f"Supplier score: {supplier_score}")