###############################################################################
###                                                                         ###
###                         NPI Sourcing simulation                         ###
###                         Saúl R. Morales © 2025                          ###
###                                                                         ###
###   This module provide the required clases to simulate the Sourcing      ###
###   NPI process. It is intended to simulate the process of sourcing       ###
###   copper piping in a new product introduction (NPI) project for HVAC    ###
###   industry, from Design Freeze (P2) to Start of Production (P3)         ###
###   stages. However, it is possible to modify this module according to    ###
###   the user's needs.                                                     ### 
###   For confidentiality purposes, statistics about prices and piping      ###
###   categories were locally stores in an .env file. An example            ###
###   containing the variables in this file is provided in .env.example.    ###
###   It is important to provide the .env file with the same variables to   ###
###   ensure the proper functionality of the module. Also, the module can   ###
###   be modified to avoid the use of .env variables.                       ###
###   If this module is used in a Google Colab Notebook, the .env           ###
###   variables can be stored as a secret.                                  ###
###                                                                         ###
###############################################################################

## Dependent modules
from datetime import date, timedelta    # To work dates.
from random import choice, random               # To generate random numbers.
import numpy as np                      # To generate random numbers.
import pandas as pd                     # To work with data frames.
import sys                              # To verify if it is running in Colab.

# Verify if the module is running in a Google Colab module.
if "google.colab" in sys.modules:               # If so, it renames getenv() to
    from google.colab import userdata           # allow using secrets without
    def getenv(secretName: str, default_value): # changing the entire code.
        try:
            return userdata.get(secretName)
        except:
            return default_value
else:                               # Else, it imports dotenv to handle .env
    from dotenv import load_dotenv  # variables.
    from os import getenv
    load_dotenv()


class Project:
   def __init__(
        self, name:str, DF_date: date, MCS_date: date,Pilot_date: date,
        SOP_date: date
    ):
      self.name = name
      self.important_dates = {
         "Design freeze": DF_date,
         "MCS": MCS_date,
         "Pilot": Pilot_date,
         "SOP": SOP_date
      }

   def __str__(self):
      return self.name

class Item_Master:
   def __init__(self):
      columns = {
        # ECN identification data
        "Project": [],
        "ECN": [],
        "ECN release": [],
        "RFQ date": [],
        # Part number data
        "Part number": [],
        "Complexity": [],
        "EAU": [],  # EAU stands for Estimated Annual Use
        # Supplier data
        "Supplier ID": [],
        "Supplier name": [],
        "Quotation date": [],
        "Price": [],
        "Lead time": [],
        # Sample delivery data
        "ETA": [],
        "Delivery date": [],
        "ISIR documents": [],
        # Environment data
        "REQ date": [],
        "PO date": [],
        "ISIR approval": [],
        "PPAP approval": [],
        "Contract date": [],
        # Fuzzy inputs
        "Quotation time": [],
        "OTD": [],
        "Delivery time": [],
        # Other information
        "FY Spend": [], # EAU * Price
        "Awarded": [], # bool (False only for quotations, True for awarded business)
        # Readiness
        "MCS ready": [],
        "Pilot ready": [],
        "SOP ready": []
      }

      self.df = pd.DataFrame(columns)

class Part_Number:
    def __init__(self, pn: str, complexity: str, eau: int):
        self.pn = pn
        self.complexity = complexity
        self.eau = eau  # I am using an integer because most of the materials have EA as UOM, with limited exceptions.

    def __str__(self):
        return self.pn

class ECN:
    def __init__(self, project: Project, ecn_id: str, ecn_date: date, pn_list: list[Part_Number]):
        self.project = project
        self.ecn_id = ecn_id
        self.ecn_date = ecn_date
        self.items = pn_list
        self.quotations = []

        self.readiness = {
            "MCS ready": False,
            "Pilot ready": False,
            "SOP ready": False
        }

    def __str__(self):
        return self.ecn_id

    def display_as_df(self):
      df_layout = {
          "Project": [],
          "ECN": [],
          "ECN release": [],
          "Part number": [],
          "Complexity": [],
          "EUA": []
       }

      df = pd.DataFrame(df_layout)

      for item in self.items:
        df.loc[len(df)] = [self.project.name, self.ecn_id, self.ecn_date, item.pn, item.complexity, item.eau]

      return df

class Quotation:
    def __init__(self, ecn: ECN, supplier: object, date: date):
        self.ecn = ecn
        self.supplier = supplier
        self.date = date
        self.awarded = False

        self.ecn.quotations.append(self)

        columns = {
        # ECN identification data
        "Project": [],
        "ECN": [],
        "ECN release": [],
        "RFQ date": [],
        # Part number data
        "Part number": [],
        "Complexity": [],
        "EAU": [],  # EAU stands for Estimated Annual Use
        # Supplier data
        "Supplier ID": [],
        "Supplier name": [],
        "Quotation date": [],
        "Price": [],
        "Lead time": [],
        # Fuzzy inputs
        "Quotation time": [],
        # Other information
        "FY Spend": [], # EAU * Price
        "Awarded": [] # bool (False only for quotations, True for awarded business)
        }

        self.df = pd.DataFrame(columns)

class Supplier:
    def __init__(self, id: str | int, name: str, price_profile: str = "regular", quotation_profile: str = "regular", punctuality_profile: str = "regular", delivery_profile: str = "regular"):
        self.id = self.__check_id(id)
        self.name = name
        self.quotations = []
        self.awarded_quotations = []

        price_profile_map = { # This is a factor to multiply; average and standard deviation
          "low": (0.85, 0.85),
          "regular": (1, 1),
          "high": (1.2, 1.1)
        }

        quotation_profile_map = { # This is a factor to multiply; average and standard deviation
          "low": (28.975, 25.1133753461483),
          "regular": (27.7241379310345, 21.5974276436511),
          "high": (24.9444444444444, 10.258266234788)
        }

        punctuality_profile_map = { # Probability
          "low": 0.19047619047619,
          "regular": 0.473684210526316,
          "high": 0.638888888888889
        }

        self.ETA_difference = {
            "punctual": (0.888888888888889, 1.01273936708367),
            "unpunctual": (4.24137931034483, 2.69463981708917)
        }

        delivery_profile_map = { # This is a factor to multiply; average and standard deviation
          "low": (0.8, 0.8),
          "regular": (1, 1),
          "high": (1, 1.3)
        }

        µ_price_profile_factor, σ_price_profile_factor = price_profile_map[price_profile]
        µ_delivey_profile_factor, σ_delivery_profile_factor = delivery_profile_map[delivery_profile]

        self.price_complexity_map = {
            "high": (float(getenv("AVG_PRICE_HIGH_COMPLEXITY", 0)) * µ_price_profile_factor, float(getenv("STDEV_PRICE_HIGH_COMPLEXITY", 1)) * σ_price_profile_factor),
            "medium": (float(getenv("AVG_PRICE_MEDIUM_COMPLEXITY", 0)) * µ_price_profile_factor, float(getenv("STDEV_PRICE_MEDIUM_COMPLEXITY", 1)) * σ_price_profile_factor),
            "low": (float(getenv("AVG_PRICE_LOW_COMPLEXITY", 0)) * µ_price_profile_factor, float(getenv("STDEV_PRICE_LOW_COMPLEXITY", 1)) * σ_price_profile_factor),
            "minimum": float(getenv("MINIMUM_PRICE", 0)) * µ_price_profile_factor
        }

        self.µ_quotation_time, self.σ_quotation_time = quotation_profile_map[quotation_profile]
        self.minimum_quotation_time = 9

        self.µ_delivery_time = 34.6206896551724 * µ_delivey_profile_factor
        self.σ_delivery_time = 16.2802512871323 * σ_delivery_profile_factor
        self.minimum_delivery_time = 12 * µ_delivey_profile_factor

        self.µ_isir_documents_upload = 0.348484848484849
        self.σ_isir_documents_upload = 0.936317241478537

        self.punctual_p = punctuality_profile_map[punctuality_profile]

    def __str__(self):
        return self.name

    def __check_id(self, id: str | int):
        if len(str(id)) < 8 or len(str(id)) > 8:
          raise Exception("Invalid supplier ID")
        else:
          return str(id)

    def quote(self, ecn: ECN, rfq_date: date, lead_time: int = 0):
        not_quoted_yet = True
        for quotation in self.quotations:
            if quotation.ecn.ecn_id == ecn.ecn_id:
                print(f"{self.name} already quoted {ecn.ecn_id}.")
                not_quoted_yet = False

        if not_quoted_yet:
          min_price = self.price_complexity_map["minimum"]

          quotation_time = max(round(np.random.normal(self.µ_quotation_time, self.σ_quotation_time)), self.minimum_quotation_time)
          quotation_date = rfq_date + timedelta(days=quotation_time)

          quotation = Quotation(ecn, self, quotation_date)

          for part_number in ecn.items:
            complexity = part_number.complexity
            µ, σ = self.price_complexity_map[complexity]
            price = round(max(np.random.normal(µ, σ), min_price), 2)
            spend = part_number.eau * price

            if lead_time == 0:
              lt = np.nan
            elif lead_time > 0:
              lt = lead_time
            else:
              raise Exception("Lead time cannot be less than 1 day.")

            quotation.df.loc[len(quotation.df)] = [ecn.project.name, ecn.ecn_id, ecn.ecn_date, rfq_date, part_number.pn, complexity, part_number.eau, self.id, self.name, quotation_date, price, lt, quotation_time, spend, False]

          self.quotations.append(quotation)
          return quotation.df