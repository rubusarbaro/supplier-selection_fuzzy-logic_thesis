"""
This module provides the required clases to simulate the Sourcing NPI process. It is intended to simulate a basic
process of sourcing in a new product introduction (NPI) project.

Classes
-------
FuzzyModel
    Provides a fuzzy logic Mamdani's model to evaluate suppliers.
"""

## Dependency modules
from logging import exception
from math import ceil, floor
from simulation import ECN, Project, Supplier
import matplotlib.pyplot as plt
import numpy as np                      # To generate random numbers.
import pandas as pd                     # To work with data frames.
import skfuzzy as fuzzy

class FuzzyModel:
    """
    Fuzzy logic model to evaluate suppliers on NPI projects.

    Attributes
    ----------
    df : DataFrame
        Pandas' data frame containing the historical data and the information of the part numbers to evaluate.
    new_supplier : bool
        Indicates if the supplier to evaluate is new (True) or existing (False).
    evaluating_supplier_id : str
        Evaluating supplier ID.
    evaluation_priority : str
        It defines the evaluation objective: "time" to prioritize implementation time. "spend" to prioritize cost reduction.
    """

    def __init__(
            self,
            df_item_master: pd.DataFrame,
            ref_supplier: Supplier,
            quotation_ecn: ECN,
            evaluation_priority: str,  # Options are: "time" and "spend"
            massive_simulation: bool = False,
            new_supplier: bool = False,
    ):
        self.df = df_item_master
        self.new_supplier = new_supplier
        self.__completely_new_supplier = False
        self.evaluating_supplier_id = ref_supplier.id
        self.evaluation_priority = evaluation_priority

        avg_delivery_time = self.df[(self.df["Awarded"] == True)]["Delivery time"].mean()  # Igual en ambos casos
        std_delivery_time = self.df[(self.df["Awarded"] == True)]["Delivery time"].std()  # Igual en ambos casos

        match self.evaluation_priority:
            case "time":
                quoted_pn = self.df[self.df["Supplier ID"] == self.evaluating_supplier_id]["Part number"]
                max_delivery_time = ceil(max(avg_delivery_time + 3 * std_delivery_time,
                                             self.df[(self.df["Awarded"] == True)]["Delivery time"].max()))
                quotations_df = self.df[self.df["Part number"].isin(quoted_pn)]
                self.spend_df = quotations_df[["Supplier ID", "FY Spend"]].groupby("Supplier ID").sum()["FY Spend"]

            case "spend":  # Revisar la obtención del spend, ya que para el objetivo 'spend' debe considerar únicamente lo del proyecto.
                self.spend_df = self.df[["Supplier ID", "FY Spend"]].groupby("Supplier ID").sum()["FY Spend"]
                max_delivery_time = ceil(self.df[(self.df["Awarded"] == True)]["Delivery time"].max())

            case _:
                exception("Evaluation priority must be 'time' or 'spend'.")

        avg_spend = (self.spend_df.mean()) / 100  # Igual en ambos casos
        std_spend = (self.spend_df.std()) / 100  # Igual en ambos casos
        min_spend = floor((self.spend_df.min()) / 100)  # Igual en ambos casos

        match evaluation_priority:
            case "time":
                if massive_simulation:
                    max_spend = (ceil(avg_spend + 10 * std_spend))
                else:
                    max_spend = (ceil(avg_spend + 3 * std_spend))
            case "spend":
                max_spend = ceil((self.spend_df.max()) / 100)

        if len(self.spend_df) < 2:
            self.__completely_new_supplier = True

        self.__var_due_time = np.arange(0, 721, 1)
        self.__var_delivery_time = np.arange(0, max_delivery_time + 1, 1)
        self.__var_spend = np.arange(0, max_spend + 1, 0.01)
        self.__var_punctuality = np.arange(0, 2, 0.01)
        self.__var_supplier = np.arange(0, 11, 0.01)

        self.__due_time_low = fuzzy.trapmf(self.__var_due_time, [0, 0, 30, 60])
        self.__due_time_medium = fuzzy.trimf(self.__var_due_time, [30, 60, 90])
        self.__due_time_high = fuzzy.trapmf(self.__var_due_time, [60, 90, 720, 720])

        self.__delivery_time_low = fuzzy.trapmf(self.__var_delivery_time,
                                                [0, 0, avg_delivery_time - std_delivery_time, avg_delivery_time])
        self.__delivery_time_medium = fuzzy.trimf(self.__var_delivery_time,
                                                  [avg_delivery_time - std_delivery_time, avg_delivery_time,
                                                 avg_delivery_time + std_delivery_time])
        self.__delivery_time_high = fuzzy.trapmf(self.__var_delivery_time,
                                                 [avg_delivery_time, avg_delivery_time + std_delivery_time,
                                                max_delivery_time, max_delivery_time])

        if not new_supplier:
            self.__punctuality_low = fuzzy.trapmf(self.__var_punctuality, [0, 0, 0.25, 0.5])
            self.__punctuality_medium = fuzzy.trimf(self.__var_punctuality, [0.25, 0.5, 0.75])
            self.__punctuality_high = fuzzy.trapmf(self.__var_punctuality, [0.5, 0.75, 1, 1])

        self.__spend_low = fuzzy.trapmf(self.__var_spend, [0, min_spend, max(min_spend, avg_spend - std_spend),
                                                           max(avg_spend - std_spend, avg_spend)])
        self.__spend_medium = fuzzy.trimf(self.__var_spend, [avg_spend - std_spend, avg_spend, avg_spend + std_spend])
        self.__spend_high = fuzzy.trapmf(self.__var_spend, [avg_spend, avg_spend + std_spend, max_spend, max_spend])

        match evaluation_priority:
            case "time":
                self.__supplier_wait = fuzzy.trapmf(self.__var_supplier, [0, 0, 5, 7.5])
                self.__supplier_implement = fuzzy.trapmf(self.__var_supplier, [2.5, 5, 10, 10])

                if not self.__completely_new_supplier:
                    self.stats = self._evaluate_supplier_time_priority(quotation_ecn)
                else:
                    self.stats = {  # Necesito actualizar este formato para poder integrar uno según el modelo.
                        "Supplier ID": ref_supplier.id,
                        "Score": 0,
                        "Wait": 1,
                        "Implement": 0,
                        "Action": "wait"
                    }

            case "spend":
                # self.__supplier_low = fuzzy.trimf(self.__var_supplier, [0, 2.5, 5])
                # self.__supplier_medium = fuzzy.trimf(self.__var_supplier, [2.5, 5, 7.5])
                # self.__supplier_high = fuzzy.trimf(self.__var_supplier, [5, 7.5, 10])

                self.__supplier_low = fuzzy.trapmf(self.__var_supplier, [0, 0, 2.5, 5])
                self.__supplier_medium = fuzzy.trimf(self.__var_supplier, [2.5, 5, 7.5])
                self.__supplier_high = fuzzy.trapmf(self.__var_supplier, [5, 7.5, 10, 10])

            case _:
                exception("Evaluation priority must be 'time' or 'spend'.")

    def get_stats(self):
        return self.stats

    def plot_model(self):
        if not self.new_supplier or self.evaluation_priority == "spend":
            fig, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(nrows=5, figsize=(8, 9))

            ax0.plot(self.__var_due_time, self.__due_time_low, "r", linewidth=1.5, label="Close")
            ax0.plot(self.__var_due_time, self.__due_time_medium, "b", linewidth=1.5, label="Near")
            ax0.plot(self.__var_due_time, self.__due_time_high, "g", linewidth=1.5, label="Far")
            ax0.set_title("Due time")
            ax0.legend()

            ax1.plot(self.__var_delivery_time, self.__delivery_time_low, "g", linewidth=1.5, label="Good")
            ax1.plot(self.__var_delivery_time, self.__delivery_time_medium, "b", linewidth=1.5, label="Regular")
            ax1.plot(self.__var_delivery_time, self.__delivery_time_high, "r", linewidth=1.5, label="Bad")
            ax1.set_title("Delivery time")
            ax1.legend()

            ax2.plot(self.__var_spend, self.__spend_low, "g", linewidth=1.5, label="Low")
            ax2.plot(self.__var_spend, self.__spend_medium, "b", linewidth=1.5, label="Regular")
            ax2.plot(self.__var_spend, self.__spend_high, "r", linewidth=1.5, label="High")
            ax2.set_title("FY Spend")
            ax2.legend()

            ax3.plot(self.__var_punctuality, self.__punctuality_low, "r", linewidth=1.5, label="Bad")
            ax3.plot(self.__var_punctuality, self.__punctuality_medium, "b", linewidth=1.5, label="Regular")
            ax3.plot(self.__var_punctuality, self.__punctuality_high, "g", linewidth=1.5, label="Good")
            ax3.set_title("Punctuality")
            ax3.legend()

            ax4.plot(self.__var_supplier, self.__supplier_wait, "r", linewidth=1.5, label="Wait")
            ax4.plot(self.__var_supplier, self.__supplier_implement, "g", linewidth=1.5, label="Implement")
            ax4.set_title("Supplier")
            ax4.legend()

            for ax in [ax0, ax1, ax2, ax3, ax4]:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()
            plt.show()

        else:
            fig, [ax0, ax1, ax2, ax3] = plt.subplots(nrows=4, figsize=(8, 9))

            ax0.plot(self.__var_due_time, self.__due_time_low, "r", linewidth=1.5, label="Close")
            ax0.plot(self.__var_due_time, self.__due_time_medium, "b", linewidth=1.5, label="Near")
            ax0.plot(self.__var_due_time, self.__due_time_high, "g", linewidth=1.5, label="Far")
            ax0.set_title("Due time")
            ax0.legend()

            ax1.plot(self.__var_delivery_time, self.__delivery_time_low, "g", linewidth=1.5, label="Good")
            ax1.plot(self.__var_delivery_time, self.__delivery_time_medium, "b", linewidth=1.5, label="Regular")
            ax1.plot(self.__var_delivery_time, self.__delivery_time_high, "r", linewidth=1.5, label="Bad")
            ax1.set_title("Delivery time")
            ax1.legend()

            ax2.plot(self.__var_spend, self.__spend_low, "g", linewidth=1.5, label="Low")
            ax2.plot(self.__var_spend, self.__spend_medium, "b", linewidth=1.5, label="Regular")
            ax2.plot(self.__var_spend, self.__spend_high, "r", linewidth=1.5, label="High")
            ax2.set_title("FY Spend")
            ax2.legend()

            ax3.plot(self.__var_supplier, self.__supplier_wait, "r", linewidth=1.5, label="Wait")
            ax3.plot(self.__var_supplier, self.__supplier_implement, "g", linewidth=1.5, label="Implement")
            ax3.set_title("Supplier")
            ax3.legend()

            for ax in [ax0, ax1, ax2, ax3]:
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()
            plt.show()

    def _evaluate_supplier_time_priority(self, quotation_ecn: ECN):
        sop_date = quotation_ecn.project.important_dates["SOP"]
        quotation_date = \
        self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["ECN"] == quotation_ecn.ecn_id)][
            "Quotation date"].max()
        due_time = sop_date - quotation_date
        crisp_due_time = max(due_time.days, 0)

        due_time_level_low = fuzzy.interp_membership(self.__var_due_time, self.__due_time_low, crisp_due_time)
        due_time_level_medium = fuzzy.interp_membership(self.__var_due_time, self.__due_time_medium, crisp_due_time)
        due_time_level_high = fuzzy.interp_membership(self.__var_due_time, self.__due_time_high, crisp_due_time)

        # Assign membership degree
        crisp_spend = (self.spend_df.loc[self.evaluating_supplier_id]) / 100

        spend_level_low = fuzzy.interp_membership(self.__var_spend, self.__spend_low, crisp_spend)
        spend_level_medium = fuzzy.interp_membership(self.__var_spend, self.__spend_medium, crisp_spend)
        spend_level_high = fuzzy.interp_membership(self.__var_spend, self.__spend_high, crisp_spend)

        if self.new_supplier:
            crisp_delivery_time = \
            self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["ECN"] == quotation_ecn.ecn_id)][
                "Lead time"].max()
        else:
            crisp_punctuality = len(self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (
                        self.df["Awarded"] == True) & (self.df["OTD"] == True)]) / len(
                self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True)])

            punctuality_level_low = fuzzy.interp_membership(self.__var_punctuality, self.__punctuality_low,
                                                            crisp_punctuality)
            punctuality_level_medium = fuzzy.interp_membership(self.__var_punctuality, self.__punctuality_medium,
                                                               crisp_punctuality)
            punctuality_level_high = fuzzy.interp_membership(self.__var_punctuality, self.__punctuality_high,
                                                             crisp_punctuality)

            crisp_delivery_time = \
            self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["ECN"] == quotation_ecn.ecn_id)][
                "Lead time"].max()

        delivery_time_level_low = fuzzy.interp_membership(self.__var_delivery_time, self.__delivery_time_low,
                                                          crisp_delivery_time)
        delivery_time_level_medium = fuzzy.interp_membership(self.__var_delivery_time, self.__delivery_time_medium,
                                                             crisp_delivery_time)
        delivery_time_level_high = fuzzy.interp_membership(self.__var_delivery_time, self.__delivery_time_high,
                                                           crisp_delivery_time)

        # Rule application
        if self.new_supplier:
            rule_1 = delivery_time_level_high  # Wait

            rule_2 = min(due_time_level_low, max(delivery_time_level_low, delivery_time_level_medium),
                         spend_level_high)  # Wait

            rule_3 = min(max(due_time_level_low, due_time_level_medium),
                         max(delivery_time_level_low, delivery_time_level_medium),
                         max(spend_level_low, spend_level_medium))  # Implement

            rule_4 = min(due_time_level_medium, spend_level_high)  # Wait

            rule_5 = min(due_time_level_high, max(delivery_time_level_low, delivery_time_level_medium),
                         spend_level_low)  # Implement

            rule_6 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high),
                         max(spend_level_medium, spend_level_high))  # Wait

            wait_strength = max(rule_1, rule_2, rule_4, rule_6)
            implement_strength = max(rule_3, rule_5)

        else:
            rule_1 = min(due_time_level_low, delivery_time_level_low, punctuality_level_low)  # Wait

            rule_2 = min(due_time_level_low, delivery_time_level_low, punctuality_level_medium,
                         spend_level_high)  # Wait

            rule_3 = min(due_time_level_low, max(delivery_time_level_medium, delivery_time_level_high),
                         max(punctuality_level_low, punctuality_level_medium))  # Wait

            rule_4 = min(due_time_level_low, delivery_time_level_medium, punctuality_level_high)  # Implement

            rule_5 = min(due_time_level_low, delivery_time_level_low, punctuality_level_medium,
                         max(spend_level_low, spend_level_medium))  # Implement

            rule_6 = min(due_time_level_low, delivery_time_level_low, punctuality_level_high)  # Implement

            rule_7 = min(due_time_level_medium, max(delivery_time_level_low, delivery_time_level_medium),
                         punctuality_level_low)  # Wait

            rule_8 = min(due_time_level_medium, max(delivery_time_level_low, delivery_time_level_medium),
                         max(punctuality_level_medium, punctuality_level_high),
                         max(spend_level_low, spend_level_medium))  # Implement

            rule_9 = min(due_time_level_medium, max(delivery_time_level_low, delivery_time_level_medium),
                         max(punctuality_level_medium, punctuality_level_high), spend_level_high)  # Wait

            rule_10 = min(due_time_level_medium, delivery_time_level_high)  # Wait

            rule_11 = min(due_time_level_high, delivery_time_level_low, spend_level_low)  # Implement

            rule_12 = min(due_time_level_high, delivery_time_level_low, punctuality_level_high,
                          spend_level_medium)  # Implement

            rule_13 = min(due_time_level_high, delivery_time_level_low,
                          max(punctuality_level_low, punctuality_level_medium),
                          max(spend_level_medium, spend_level_high))  # Wait

            rule_14 = min(due_time_level_high, delivery_time_level_low, punctuality_level_high,
                          spend_level_high)  # Wait

            rule_15 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high),
                          punctuality_level_low, spend_level_medium)  # Wait

            rule_16 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high),
                          spend_level_high)  # Wait

            rule_17 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high),
                          punctuality_level_low, spend_level_low)  # Implement

            rule_18 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high),
                          max(punctuality_level_medium, punctuality_level_high),
                          max(spend_level_low, spend_level_medium))  # Implement

            wait_strength = max(rule_1, rule_2, rule_3, rule_7, rule_9, rule_10, rule_13, rule_14, rule_15, rule_16)
            implement_strength = max(rule_4, rule_5, rule_6, rule_8, rule_11, rule_12, rule_17, rule_18)

        supplier_activation_wait = np.fmin(wait_strength, self.__supplier_wait)
        supplier_activation_implement = np.fmin(implement_strength, self.__supplier_implement)

        aggregated = np.fmax.reduce([supplier_activation_wait, supplier_activation_implement])

        # Defuzzification
        supplier_score = fuzzy.defuzz(self.__var_supplier, aggregated, "centroid")
        supplier_activation = fuzzy.interp_membership(self.__var_supplier, aggregated, supplier_score)

        if supplier_activation_implement.max() > supplier_activation_wait.max():
            action = "Implement"
        else:
            action = "Wait"

        stats = {
            "Supplier ID": self.evaluating_supplier_id,
            "New supplier": self.new_supplier,
            "Score": supplier_score,
            "Wait": supplier_activation_wait.max(),
            "Implement": supplier_activation_implement.max(),
            "Action": action
        }

        if self.new_supplier:
            rule_number = 1
            for rule in [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6]:
                stats[f"Rule {rule_number}"] = rule
                rule_number += 1
            for i in range(7, 19):
                stats[f"Rule {i}"] = np.nan
        else:
            rule_number = 1
            for rule in [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10, rule_11,
                         rule_12, rule_13, rule_14, rule_15, rule_16, rule_17, rule_18]:
                stats[f"Rule {rule_number}"] = rule
                rule_number += 1

        return stats

    def _evaluate_supplier_spend_priority(self, project: Project, gen_chart: bool = False):
        crisp_price = \
        self.df[(self.df["Project"] == project.name) & (self.df["Supplier ID"] == self.evaluating_supplier_id)][
            "FY Spend"].sum() / 100

        if self.new_supplier:
            crisp_delivery_time = self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id)]["Lead time"].mean()

        else:
            crisp_punctuality = len(self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (
                        self.df["Awarded"] == True) & (self.df["OTD"] == True)]) / len(
                self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True)])
            crisp_delivery_time = \
            self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True)][
                "Delivery time"].mean()

            punctuality_level_low = fuzzy.interp_membership(self.__var_punctuality, self.__punctuality_low,
                                                            crisp_punctuality)
            punctuality_level_medium = fuzzy.interp_membership(self.__var_punctuality, self.__punctuality_medium,
                                                               crisp_punctuality)
            punctuality_level_high = fuzzy.interp_membership(self.__var_punctuality, self.__punctuality_high,
                                                             crisp_punctuality)

        # Assign membership degree
        price_level_low = fuzzy.interp_membership(self.__var_spend, self.__spend_low, crisp_price)
        price_level_medium = fuzzy.interp_membership(self.__var_spend, self.__spend_medium, crisp_price)
        price_level_high = fuzzy.interp_membership(self.__var_spend, self.__spend_high, crisp_price)

        delivery_time_level_low = fuzzy.interp_membership(self.__var_delivery_time, self.__delivery_time_low,
                                                          crisp_delivery_time)
        delivery_time_level_medium = fuzzy.interp_membership(self.__var_delivery_time, self.__delivery_time_medium,
                                                             crisp_delivery_time)
        delivery_time_level_high = fuzzy.interp_membership(self.__var_delivery_time, self.__delivery_time_high,
                                                           crisp_delivery_time)

        # Rule application
        # Example code uses np.fmax for OR operator. I will use np.fmin for AND.
        if self.new_supplier:
            rule_1 = min(max(price_level_low, price_level_medium), delivery_time_level_low)  # High

            rule_2 = min(max(price_level_low, price_level_medium), delivery_time_level_medium)  # Regular

            rule_3 = min(max(price_level_low, price_level_medium), delivery_time_level_high)  # Low

            rule_4 = price_level_high  # Low

            low_strength = max(rule_3, rule_4)
            medium_strength = rule_2
            high_strength = rule_1
        else:
            rule_1 = min(delivery_time_level_low, max(price_level_low, price_level_medium),
                         punctuality_level_low)  # Regular

            rule_2 = min(delivery_time_level_low, max(price_level_low, price_level_medium),
                         max(punctuality_level_medium, punctuality_level_high))  # High

            rule_3 = min(delivery_time_level_low, price_level_high, punctuality_level_low)  # Low

            rule_4 = min(delivery_time_level_low, price_level_high,
                         max(punctuality_level_medium, punctuality_level_high))  # Regular

            rule_5 = min(delivery_time_level_medium, price_level_low, punctuality_level_low)  # Regular

            rule_6 = min(max(delivery_time_level_medium, delivery_time_level_high),
                         max(price_level_low, price_level_medium), punctuality_level_medium)  # Regular

            rule_7 = min(max(delivery_time_level_medium, delivery_time_level_high),
                         max(price_level_low, price_level_medium), punctuality_level_high)  # High

            rule_8 = min(delivery_time_level_medium, max(price_level_medium, price_level_high),
                         punctuality_level_low)  # Low

            rule_9 = min(max(delivery_time_level_medium, delivery_time_level_high), price_level_high,
                         punctuality_level_medium)  # Low

            rule_10 = min(max(delivery_time_level_medium, delivery_time_level_high), price_level_high,
                          punctuality_level_high)  # Medium

            rule_11 = min(delivery_time_level_high, punctuality_level_low)  # Low

            low_strength = max(rule_3, rule_8, rule_9, rule_11)
            medium_strength = max(rule_1, rule_4, rule_5, rule_6, rule_10)
            high_strength = max(rule_2, rule_7)

        supplier_activation_low = np.fmin(low_strength, self.__supplier_low)
        supplier_activation_medium = np.fmin(medium_strength, self.__supplier_medium)
        supplier_activation_high = np.fmin(high_strength, self.__supplier_high)

        supplier_0 = np.zeros_like(self.__var_supplier)

        aggregated = np.fmax.reduce([supplier_activation_low, supplier_activation_medium, supplier_activation_high])

        # Defuzzification
        supplier_score = fuzzy.defuzz(self.__var_supplier, aggregated, "centroid")
        supplier_activation = fuzzy.interp_membership(self.__var_supplier, aggregated, supplier_score)

        max_activation = max(
            supplier_activation_low.max(),
            supplier_activation_medium.max(),
            supplier_activation_high.max()
        )

        if max_activation == supplier_activation_low.max():
            linguistic_tag = "Low"
        elif max_activation == supplier_activation_medium.max():
            linguistic_tag = "Regular"
        else:
            linguistic_tag = "High"

        stats = {
            "Supplier ID": self.evaluating_supplier_id,
            "New supplier": self.new_supplier,
            "Score": supplier_score,
            "Low": supplier_activation_low.max(),
            "Regular": supplier_activation_medium.max(),
            "High": supplier_activation_high.max(),
            "Classification": linguistic_tag
        }

        if self.new_supplier:
            rule_number = 1
            for rule in [rule_1, rule_2, rule_3, rule_4]:
                stats[f"Rule {rule_number}"] = rule
                rule_number += 1
            for i in range(5, 12):
                stats[f"Rule {i}"] = np.nan
        else:
            rule_number = 1
            for rule in [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10, rule_11]:
                stats[f"Rule {rule_number}"] = rule
                rule_number += 1

        if gen_chart:
            fig, ax0 = plt.subplots(figsize=(8, 3))

            ax0.fill_between(self.__var_supplier, supplier_0, supplier_activation_low, facecolor="r", alpha=0.7)
            ax0.plot(self.__var_supplier, self.__supplier_low, "r", linewidth=0.5, linestyle="--", )
            ax0.fill_between(self.__var_supplier, supplier_0, supplier_activation_medium, facecolor="b", alpha=0.7)
            ax0.plot(self.__var_supplier, self.__supplier_medium, "b", linewidth=0.5, linestyle="--", )
            ax0.fill_between(self.__var_supplier, supplier_0, supplier_activation_high, facecolor="g", alpha=0.7)
            ax0.plot(self.__var_supplier, self.__supplier_high, "g", linewidth=0.5, linestyle="--", )
            ax0.set_title(f"Output membership activity (Supplier {self.evaluating_supplier_id})")

            for ax in (ax0,):
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.get_xaxis().tick_bottom()
                ax.get_yaxis().tick_left()

            plt.tight_layout()
            plt.show()

        return stats