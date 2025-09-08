###############################################################################
###                                                                         ###
###                         NPI Sourcing simulation                         ###
###                    Saúl R. Morales © 2025 MIT License                   ###
###                                                                         ###
###   This module provides the required clases to simulate the Sourcing     ###
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
"""
This module provides the required clases to simulate the Sourcing NPI process. It is intended to simulate a basic process of sourcing in a new product introduction (NPI) project.

Classes
-------
ECN
  Represents an ECN (Engineering Change Notification) as a set of part numbers.
Environment
  Represents the environment were the NPI project takes place and objects interact.
Item_Master
  Provides a Pandas' DataFrame with a standardized column set.
Part_Number
  Represents a single material or part number.
Project
  Represents a NPI (New Product Introduction) project.
Quotation
  Represents a supplier's quotation.
Supplier
  Represents a supplier.
"""

## Dependent modules
from datetime import date, timedelta    # To work dates.
from math import ceil, floor
from statistics import mean, stdev
from random import choice, random       # To generate random numbers.
import matplotlib.pyplot as plt
import colors
import misc
import numpy as np                      # To generate random numbers.
import pandas as pd                     # To work with data frames.
import skfuzzy as fuzzy
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
  """
  Represents a NPI project.

  Attributes
  ----------
  name : str
    Name of NPI project.
  ecns : list[ECN]
    List object containing the ECNs related to the project.
  important_dates : dict[str, date]
    Dictionary containing the important dates for the project, such as Design Freeze, MCS, Pilot and SOP.
  
  Methods
  -------
  """
  def __init__(
    self,
    name: str,
    df_date: date,
    mcs_date: date, 
    pilot_date: date,
    sop_date: date):
      """
      Initialize Project object.

      Parameters:
        name (str): Name of NPI project.
        df_date (date): Date of design freeze. Design freeze is the date when non-critical changes to the project cease. ECNs are released near this date.
        mcs_date (date): Date of MCS. Samples need to arrive before this date, but they do not need to be PPAP approved.
        pilot_date (date): Date of pilot. Building pilot units. Material need to be PPAP approved and in SAP contract before this date; procurement is made by regular purchasing process.
        sop_date (date): Date of start of production. Implementation of NPI project.
      """
      self.name = name
      self.ecns = []
      self.important_dates = {
         "Design freeze": df_date,
         "MCS": mcs_date,
         "Pilot": pilot_date,
         "SOP": sop_date
      }

  def __str__(self):
      """Returns project name when the object is printed."""
      return self.name

class Item_Master:
   """
   Standarized column format for the data frame.

   Attributes
   ----------
   df : DataFrame
      Pandas' data frame.
   """

   def __init__(self):
      """Initialize the Item Master object."""
      columns = {
        # ECN identification data
        "Project": [],
        "ECN": [],
        "ECN release": [],
        # Part number data
        "Part number": [],
        "Complexity": [],
        "EAU": [],  # EAU stands for Estimated Annual Use
        # Supplier data
        "Delivery profile": [],
        "Quotation profile": [],
        "Price profile": [],
        "Punctuality profile": [],
        "Supplier ID": [],
        "Supplier name": [],
        "Price": [],
        "Lead time": [],
        # Dates
        "RFQ date": [],
        "Quotation date": [],
        "REQ date": [],
        "PO date": [],
        "ETA": [],
        "Delivery date": [],
        "ISIR documents": [],        
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
    """
    Represents a single material or part number.
    This class is made to represent exclusively copper pipes. However, it could be used for any kind of material.

    Attributes
    ----------
    pn : str
      Unique part number for the material. There must not be two materials with the same part number.
    complexity : str
      Complexity of the copper pipe.
    eau : int
      Estimated Annual Use, is the quantity of the material that is intended to consume during the fiscal year.
    """
    def __init__(self, pn: str, complexity: str, eau: int):
        """
        Initialize the Part_Number object.

        Parameters:
          pn (str): Unique part number for the material.
          complexity (str): Complexity of the material. Valid options are 'low', 'medium', 'high'.
          eau (int): Estimated Annual Use, is the quantity of the material that is intended to consume during the fiscal year.
        """
        self.pn = pn
        self.complexity = complexity
        self.eau = eau  # I am using an integer because most of the materials have EA as UOM, with limited exceptions.

    def __str__(self):
        """Returns part number identifier when the object is printed."""
        return self.pn

class ECN:
    """
    Represents an ECN (Engineering Change Notification) as a set of part numbers.

    Attributes
    ----------
    instances : int
      Counter of ECN objects.
    project : Project
      Project object related to this ECN.
    ecn_id : str
      ECN unique identifier; autogenerated.
    ecn_date : date
      Date when ECN was released.
    items : list[Part_Number]
      List of part numbers released in this ECN.
    quotations : list[Quotation]
      List of quotations related to this ECN.
    readiness : dict[str, bool]
      Dictionary containing bools (True or False) indicating id the ECN implementation met the dates of the project.

    Methods
    -------
    display_as_df():
      Returns a DataFrame containing the columns 'Project', 'ECN', 'ECN release', 'Part number', 'Complexity', 'EAU', and its values.
    """
    instances = 0

    def __init__(self, project: Project, ecn_date: date, pn_list: list[Part_Number]):
        """
        Initialize the ECN object.

        Parameters:
          project (Project): Project related to this ECN.
          ecn_date (date): Date when ECN was released in date format.
          pn_list (list[Part_Number]): List of part numbers released in this ECN.
        """
        ECN.instances += 1
        self.project = project
        self.ecn_id = f"ECN{str(ECN.instances).zfill(7)}"
        self.ecn_date = ecn_date
        self.items = pn_list
        self.quotations = []

        self.readiness = {
            "MCS ready": False,
            "Pilot ready": False,
            "SOP ready": False
        }

    def __str__(self):
        """ECN id when the object is printed."""
        return self.ecn_id

    def display_as_df(self):
      """Returns ECN as a Pandas DataFrame, containing the information for each part number in a row."""
      df_layout = {
          "Project": [],
          "ECN": [],
          "ECN release": [],
          "Part number": [],
          "Complexity": [],
          "EAU": []
       }

      df = pd.DataFrame(df_layout)

      for item in self.items:
        df.loc[len(df)] = [self.project.name, self.ecn_id, self.ecn_date, item.pn, item.complexity, item.eau]

      return df

class Quotation:
    """
    Represents a supplier's quotation.

    Attributes
    ----------
    ecn : ECN
      ECN related to this quotation.
    supplier : Supplier
      Supplier who made this quotation.
    date : date
      Quotation date.
    awarded : bool
      Indicates if the quotation (supplier) was awarded with the business.
    df : DataFrame
      Quotation as Pandas DataFrame.
    """
    def __init__(self, ecn: ECN, supplier: object, date: date):
        """
        Initializes the Quotation object.

        Parameters:
          ecn (ECN): ECN object related to this quotation.
          supplier (Supplier): Supplier object who made this quotation.
          date (date): Quotation date in date format.
        """
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
        # Part number data
        "Part number": [],
        "Complexity": [],
        "EAU": [],  # EAU stands for Estimated Annual Use
        # Supplier data
        "Delivery profile": [],
        "Quotation profile": [],
        "Price profile": [],
        "Punctuality profile": [],
        "Supplier ID": [],
        "Supplier name": [],
        "Price": [],
        "Lead time": [],
        # Dates
        "RFQ date": [],
        "Quotation date": [],
        # Fuzzy inputs
        "Quotation time": [],
        # Other information
        "FY Spend": [], # EAU * Price
        "Awarded": [], # bool (False only for quotations, True for awarded business)
      }

        self.df = pd.DataFrame(columns)

class Supplier:
    """
    Represents a supplier.

    Attributes
    ----------
    instances : int
    id : str
    name : str
    quotations : list[Quotation]
    awarded_ecns : list[ECN]
    delivery_profile : str
    quotation_profile : str
    price_profile : str
    punctuality_profile : str

    """
    instances = 0

    # Factor dictionaries
    __delivery_profile_factor = { # This is a factor to multiply; average and standard deviation
      "low": (0.8, 0.8),
      "regular": (1, 1),
      "high": (1, 1.3)
    }

    __quotation_profile_factor = { # This is a factor to multiply; average and standard deviation
      "low": (0.8, 0.8),
      "regular": (1, 1),
      "high": (1.3, 1.1)
    }

    __price_profile_factor = { # This is a factor to multiply; average and standard deviation
      "low": (0.85, 0.85),
      "regular": (1, 1),
      "high": (1.2, 1.1)
    }

    def __init__(self, name: str, delivery_profile: str = "regular", quotation_profile: str = "regular", price_profile: str = "regular", punctuality_profile: str = "regular", standard_lt: int = 0):
        Supplier.instances += 1
        self.id = f"1{str(Supplier.instances).zfill(7)}"
        self.name = name
        self.quotations = []
        self.awarded_ecns = []
        self.standard_lead_time = standard_lt

        self.delivery_profile = delivery_profile
        self.quotation_profile = quotation_profile
        self.price_profile = price_profile
        self.punctuality_profile = punctuality_profile

        # Factor definition (average and stdev)
        µ_delivery_profile_factor, σ_delivery_profile_factor = Supplier.__delivery_profile_factor[delivery_profile]
        µ_quotation_profile_factor, σ_quotation_profile_factor = Supplier.__quotation_profile_factor[quotation_profile]
        µ_price_profile_factor, σ_price_profile_factor = Supplier.__price_profile_factor[price_profile]

        # Averages and standard deviation
        self.µ_delivery_time = 34.6206896551724 * µ_delivery_profile_factor
        self.σ_delivery_time = 16.2802512871323 * σ_delivery_profile_factor
        self.minimum_delivery_time = 12 * µ_delivery_profile_factor

        self.eta_difference = {
            "punctual": (0.888888888888889, 1.01273936708367),
            "unpunctual": (4.24137931034483, 2.69463981708917)
        }

        self.µ_isir_documents_upload = 0.348484848484849
        self.σ_isir_documents_upload = 0.936317241478537

        self.µ_quotation_time = 27.7241379310345 * µ_quotation_profile_factor
        self.σ_quotation_time = 21.5974276436511 * σ_quotation_profile_factor
        self.minimum_quotation_time = 9 * µ_quotation_profile_factor

        self.price_complexity_map = {
            "high": (float(getenv("AVG_PRICE_HIGH_COMPLEXITY", 0)) * µ_price_profile_factor, float(getenv("STDEV_PRICE_HIGH_COMPLEXITY", 1)) * σ_price_profile_factor),
            "medium": (float(getenv("AVG_PRICE_MEDIUM_COMPLEXITY", 0)) * µ_price_profile_factor, float(getenv("STDEV_PRICE_MEDIUM_COMPLEXITY", 1)) * σ_price_profile_factor),
            "low": (float(getenv("AVG_PRICE_LOW_COMPLEXITY", 0)) * µ_price_profile_factor, float(getenv("STDEV_PRICE_LOW_COMPLEXITY", 1)) * σ_price_profile_factor),
            "minimum": float(getenv("MINIMUM_PRICE", 0)) * µ_price_profile_factor
        }

        # Probabilities
        punctuality_profile_map = { # Probability
          "low": 0.19047619047619,
          "regular": 0.473684210526316,
          "high": 0.638888888888889
        }

        self.punctual_p = punctuality_profile_map[punctuality_profile]

    def __str__(self):
        return self.name

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

            if lead_time == 0 and self.standard_lead_time == 0:
              lt = np.nan
            elif lead_time > 0:
              lt = lead_time
            else:
              lt = self.standard_lead_time

            quotation.df.loc[len(quotation.df)] = [
              ecn.project.name,
              ecn.ecn_id,
              ecn.ecn_date,
              part_number.pn,
              complexity,
              part_number.eau,
              self.delivery_profile,
              self.quotation_profile,
              self.price_profile,
              self.punctuality_profile,
              self.id,
              self.name,
              price,
              lt,
              rfq_date,
              quotation_date,
              quotation_time,
              spend,
              False
            ]

          self.quotations.append(quotation)
          return quotation.df
        
class Environment:
  def __init__(self):
    self.suppliers = []
    self.active_suppliers = []
    self.inactive_suppliers = []
    self.ecns = []

    self.item_master = Item_Master().df

    self.part_kinds = {
        "A": {
            "average": float(getenv("AVG_A_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_A_PART_KIND", 1)),
            "complexity": {"low": 0.6818181818182, "medium": 0.318181818181818, "high": 0},
            "parts": []
        },
        "B": {
            "average": float(getenv("AVG_B_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_B_PART_KIND", 1)),
            "complexity": {"low": 1/3, "medium": 2/3, "high": 0},
            "parts": []
        },
        "C": {
            "average": float(getenv("AVG_C_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_C_PART_KIND", 1)),
            "complexity": {"low": 1, "medium": 0, "high": 0},
            "parts": []
        },
        "D": {
            "average": float(getenv("AVG_D_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_D_PART_KIND", 1)),
            "complexity": {"low": 0.090909090909090909, "medium": 0.727272727272727, "high": 0.181818181818182},
            "parts": []
        },
        "E": {
            "average": float(getenv("AVG_E_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_E_PART_KIND", 1)),
            "complexity": {"low": 0, "medium": 0, "high": 1},
            "parts": []
        },
        "F": {
            "average": float(getenv("AVG_F_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_F_PART_KIND", 1)),
            "complexity": {"low": 0, "medium": 0, "high": 1},
            "parts": []
        },
        "G": {
            "average": float(getenv("AVG_G_PART_KIND", 0)),
            "stdev": float(getenv("STDEV_G_PART_KIND", 1)),
            "complexity": {"low": 0, "medium": 0, "high": 1},
            "parts": []
        }
    }

    self.environment_times = {
      "release_ecn": (119.47619047619*0.5, 160.671596446795*0.5625),
      "release_ecn_min": -110,
      "release_ecn_max": 436,
      "send_rfq": (3.0625, 3.98415201798981),
      "create_req": (8.88571428571429, 8.92406428682835),
      "send_po": (7.53409090909091, 3.31471419560376),
      "approve_isir": (26.8, 33.7110446645324),
      "approve_ppap": (2.63559322033898, 5.29741955281872),
      "upload_contract": (2.2156862745098, 3.23234718683581)
    }

    self.µ_eau_qty = 319.395833333333
    self.σ_eau_qty = 364.965095363013
    self.min_eau_qty = 23

  def add_supplier(self, supplier: Supplier):
    self.suppliers.append(supplier)
    self.active_suppliers.append(supplier)

  def add_suppliers(self, suppliers: list[Supplier]):
    for supplier in suppliers:
      self.add_supplier(supplier)

  def activate_supplier(self, supplier: Supplier):
      if supplier in self.active_suppliers:
        print(f"{supplier.name} is already active.")
      else:
        self.active_suppliers.append(supplier)
        self.inactive_suppliers.remove(supplier)
  
  def create_supplier(self, name: str, delivery_profile: str = "regular", quotation_profile: str = "regular", price_profile: str = "regular", punctuality_profile: str = "regular", active: bool = True):
    for supplier in self.suppliers:
      if supplier.name == name:
        raise Exception(f"Supplier {name} already exists. Its ID is {supplier.id}.")

    supplier = Supplier(name, delivery_profile, quotation_profile, price_profile, punctuality_profile)
    self.add_supplier(supplier)

    if not active:
      self.deactivate_supplier(supplier)
  
  def deactivate_supplier(self, supplier: Supplier):
    self.inactive_suppliers.append(supplier)
    self.active_suppliers.remove(supplier)

  def gen_ecns(self, project: Project, qty: int):
    ecns = []

    for i in range(qty):
      ecn_part_numbers = []
      ecn_eau = max(round(np.random.normal(loc=self.μ_eau_qty, scale=self.σ_eau_qty)), self.min_eau_qty)

      while len(ecn_part_numbers) == 0:
        for key in self.part_kinds.keys():
          kind_complexity_keys = list(self.part_kinds[key]["complexity"].keys())
          kind_complexity_probabilities = list(self.part_kinds[key]["complexity"].values())

          for j in range(max(int(np.random.normal(self.part_kinds[key]["average"], self.part_kinds[key]["stdev"])), 0)):
            category_part_number = len(self.part_kinds[key]["parts"]) + 1
            complexity = np.random.choice(kind_complexity_keys, p=kind_complexity_probabilities)

            part_number = Part_Number(pn=f"A0{key}{str(category_part_number).zfill(6)}", complexity=complexity, eau=ecn_eau)

            self.part_kinds[key]["parts"].append(part_number)
            ecn_part_numbers.append(part_number)

      µ_ecn_release_time, σ_ecn_release_time = self.environment_times["release_ecn"]
      ecn_date = project.important_dates["Design freeze"] + timedelta(days=min(max(round(np.random.normal(loc=µ_ecn_release_time, scale=σ_ecn_release_time)), -160), 436))

      ecn = ECN(project=project, ecn_date=ecn_date, pn_list=ecn_part_numbers)
      self.ecns.append(ecn)
      project.ecns.append(ecn)
      ecns.append(ecn)

    return ecns

  def get_ecn(self, search_reference: str):
     for ecn in self.ecns:
        if ecn.ecn_id == search_reference:
           return ecn
  
  def get_supplier(self, search_mode: str, reference: str):
    for supplier in self.suppliers:
      match search_mode:
          case "name":
            if supplier.name == reference:
              return supplier
          case "id":
            if len(reference) != 8:
              raise Exception("Supplier ID length must be 8 characters.")
            if supplier.id == reference:
              return supplier

  def quote_all_ecn_project_all_suppliers(self, project: Project):
     for ecn in project.ecns:
        self.quote_ecn_all_suppliers(ecn)

  def quote_ecn_all_suppliers(self, ecn: ECN):
    µ_rfq_time, σ_rfq_time = self.environment_times["send_rfq"]

    for supplier in self.active_suppliers:
      rfq_date = ecn.ecn_date + timedelta(days=max(round(np.random.normal(loc=µ_rfq_time, scale=σ_rfq_time)), 0))

      quotation = supplier.quote(ecn, rfq_date)
      self.item_master = pd.concat([self.item_master, quotation], ignore_index=True)

    return self.item_master[self.item_master["ECN"] == ecn.ecn_id]
  
  def quote_ecn_some_suppliers(self, ecn: ECN, quoting_suppliers: list[Supplier]):
    µ_rfq_time, σ_rfq_time = self.environment_times["send_rfq"]

    for supplier in quoting_suppliers:
      if supplier in self.active_suppliers:
          rfq_date = ecn.ecn_date + timedelta(days=max(round(np.random.normal(loc=µ_rfq_time, scale=σ_rfq_time)), 0))

          quotation = supplier.quote(ecn, rfq_date)
          self.item_master = pd.concat([self.item_master, quotation], ignore_index=True)

          return self.item_master[self.item_master["ECN"] == ecn.ecn_id]
      else:
          print(f"{supplier.name} is inactive.")

  def implement_ecn(self, ecn: ECN, awarded_supplier: Supplier, overwrite: bool = False):
    if not overwrite:
      for supplier in self.suppliers:
        if supplier != awarded_supplier:
          for quotation in supplier.quotations:
            if quotation.ecn == ecn and quotation.awarded == True:
              raise Exception(f"{ecn.ecn_id} has already been implemented with {supplier.name}.")

      quotation_not_found = True
      working_quotation = None
      for quotation in awarded_supplier.quotations:
        if quotation.ecn == ecn:
          if quotation.awarded == True:
            raise Exception(f"{ecn.ecn_id} has already been implemented with {awarded_supplier.name}.")
          else:
            quotation.awarded = True
            working_quotation = quotation
            quotation_not_found = False
      if quotation_not_found:
        raise Exception(f"{ecn.ecn_id} has not been quoted by {awarded_supplier.name}.")
    else:
        quotation_not_found = True
        working_quotation = None
        for quotation in awarded_supplier.quotations:
            if quotation.ecn == ecn:
              quotation.awarded = True
              working_quotation = quotation
              quotation_not_found = False
        if quotation_not_found:
            raise Exception(f"{ecn.ecn_id} has not been quoted by {awarded_supplier.name}.")

    µ_req_time, σ_req_time = self.environment_times["create_req"]
    µ_po_time, σ_po_time = self.environment_times["send_po"]
    µ_delivery_time, σ_delivery_time = (awarded_supplier.μ_delivery_time, awarded_supplier.σ_delivery_time)
    µ_documents_time, σ_documents_time = (awarded_supplier.μ_isir_documents_upload, awarded_supplier.σ_isir_documents_upload)
    µ_isir_time, σ_isir_time = self.environment_times["approve_isir"]
    µ_ppap_time, σ_ppap_time = self.environment_times["approve_ppap"]
    µ_contract_time, σ_contract_time = self.environment_times["upload_contract"]

    req_time = max(round(np.random.normal(loc=µ_req_time, scale=σ_req_time)), 0)
    po_time = max(round(np.random.normal(loc=µ_po_time, scale=σ_po_time)), 0)
    eta_time = max(round(np.random.normal(loc=µ_delivery_time, scale=σ_delivery_time)), awarded_supplier.minimum_delivery_time)

    if random() < awarded_supplier.punctual_p:
      µ_eta_difference, σ_eta_difference = awarded_supplier.eta_difference["punctual"]
      delivery_time = -max(round(np.random.normal(loc=µ_eta_difference, scale=σ_eta_difference)), 0)
      otd = True
    else:
      µ_eta_difference, σ_eta_difference = awarded_supplier.eta_difference["unpunctual"]
      delivery_time = max(round(np.random.normal(loc=µ_eta_difference, scale=σ_eta_difference)), 1)
      otd = False

    documents_time = max(round(np.random.normal(loc=µ_documents_time, scale=σ_documents_time)), 0)
    isir_time = max(round(np.random.normal(loc=µ_isir_time, scale=σ_isir_time)), 0)
    ppap_time = max(round(np.random.normal(loc=µ_ppap_time, scale=σ_ppap_time)), 0)
    contract_time = max(round(np.random.normal(loc=µ_contract_time, scale=σ_contract_time)), 0)

    req_date = working_quotation.date + timedelta(days=req_time)
    po_date = req_date + timedelta(days=po_time)
    eta_date = po_date + timedelta(days=eta_time)
    delivery_date = eta_date + timedelta(days=delivery_time)
    documents_date = delivery_date + timedelta(days=documents_time)
    isir_date = documents_date + timedelta(days=isir_time)
    ppap_date = isir_date + timedelta(days=ppap_time)
    contract_date = ppap_date + timedelta(days=contract_time)

    if delivery_date <= ecn.project.important_dates["MCS"]:
      mcs_ready = True
    else:
      mcs_ready = False

    if ppap_date <= ecn.project.important_dates["Pilot"]:
      pilot_ready = True
    else:
      pilot_ready = False

    if contract_date <= ecn.project.important_dates["SOP"] - timedelta(weeks=6):
      sop_ready = True
    else:
      sop_ready = False

    self.item_master.loc[
      (self.item_master["ECN"] == ecn.ecn_id) & (self.item_master["Supplier ID"] == awarded_supplier.id),
      [
        "REQ date",
        "PO date",
        "ETA",
        "Delivery date",
        "ISIR documents",
        "ISIR approval",
        "PPAP approval",
        "Contract date",
        "OTD",
        "Delivery time",
        "Awarded",
        "MCS ready",
        "Pilot ready",
        "SOP ready"
      ]] = [
        req_date,
        po_date,
        eta_date,
        delivery_date,
        documents_date,
        isir_date,
        ppap_date,
        contract_date,
        otd,
        eta_time + delivery_time,
        True,
        mcs_ready,
        pilot_ready,
        sop_ready
      ]

    awarded_supplier.awarded_ecns.append(ecn)
    return self.item_master[(self.item_master["ECN"] == ecn.ecn_id) & (self.item_master["Supplier ID"] == awarded_supplier.id)]

  def quote_all_ecns(self):
    for ecn in self.ecns:
      self.quote_ecn_all_suppliers(ecn)
    return self.item_master

  def gen_initial_item_master_df(self):
    for ecn in self.ecns:
      random_supplier = choice(self.suppliers)

      self.implement_ecn(ecn, random_supplier)
    
    return self.item_master
  
  def gen_initial_item_master_df_project(self, project: Project):
    for ecn in project.ecns:
      random_supplier = choice(self.suppliers)

      self.implement_ecn(ecn, random_supplier)
    
    return self.item_master
  
  def get_µσ_punctuality(self):
    punctuality_list = []

    for supplier_id in self.item_master[(self.item_master["Awarded"] == True)]["Supplier ID"].unique():
        punctuality = len(self.item_master[(self.item_master["Supplier ID"] == supplier_id) & (self.item_master["Awarded"] == True) & (self.item_master["OTD"] == True)]) / len(self.item_master[(self.item_master["Supplier ID"] == supplier_id) & (self.item_master["Awarded"] == True)])

        punctuality_list.append(punctuality)

    return (mean(punctuality_list), stdev(punctuality_list))
  
class Fuzzy_Model:
    def __init__(self, df_item_master: pd.DataFrame):
      self.df = df_item_master

      avg_delivery_time = self.df[(self.df["Awarded"] == True)]["Delivery time"].mean()
      std_delivery_time = self.df[(self.df["Awarded"] == True)]["Delivery time"].std()

      avg_price = self.df[["Supplier name", "Price"]].groupby("Supplier name").sum()["Price"].mean()
      std_price = self.df[["Supplier name", "Price"]].groupby("Supplier name").sum()["Price"].std()
      
      max_delivery_time = ceil(avg_delivery_time + 3*std_delivery_time)

      min_price = min(floor(avg_price - 3*std_price), 0)
      max_price = ceil(avg_price + 3*std_price)

      self.var_delivery_time = np.arange(0, max_delivery_time + 1, 1)
      self.var_price = np.arange(min_price, max_price + 1, 0.01)
      self.var_punctuality = np.arange(0, 2, 0.01)
      self.var_supplier = np.arange(0, 11, 0.01)

      self.delivery_time_low = fuzzy.trapmf(self.var_delivery_time, [0, 0, avg_delivery_time - std_delivery_time, avg_delivery_time])
      self.delivery_time_medium = fuzzy.trimf(self.var_delivery_time, [avg_delivery_time - std_delivery_time, avg_delivery_time, avg_delivery_time + std_delivery_time])
      self.delivery_time_high = fuzzy.trapmf(self.var_delivery_time, [avg_delivery_time, avg_delivery_time + std_delivery_time, max_delivery_time, max_delivery_time])

      self.punctuality_low = fuzzy.trapmf(self.var_punctuality, [0, 0, 0.25, 0.5])
      self.punctuality_medium = fuzzy.trimf(self.var_punctuality, [0.25, 0.5, 0.75])
      self.punctuality_high = fuzzy.trapmf(self.var_punctuality, [0.5, 0.75, 1, 1])

      self.price_low = fuzzy.trapmf(self.var_price, [0, min_price, avg_price - std_price, avg_price])
      self.price_medium = fuzzy.trimf(self.var_price, [avg_price - std_price, avg_price, avg_price + std_price])
      self.price_high = fuzzy.trapmf(self.var_price, [avg_price, avg_price + std_price, max_price, max_price])

      self.supplier_wait = fuzzy.trapmf(self.var_supplier, [0, 0, 5, 7.5])
      self.supplier_implement = fuzzy.trapmf(self.var_supplier, [2.5, 5, 10, 10])

    def plot(self):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

        ax0.plot(self.var_delivery_time, self.delivery_time_low, "g", linewidth=1.5, label="Good")
        ax0.plot(self.var_delivery_time, self.delivery_time_medium, "b", linewidth=1.5, label="Regular")
        ax0.plot(self.var_delivery_time, self.delivery_time_high, "r", linewidth=1.5, label="Bad")
        ax0.set_title("Delivery time")
        ax0.legend()

        ax1.plot(self.var_punctuality, self.punctuality_low, "r", linewidth=1.5, label="Bad")
        ax1.plot(self.var_punctuality, self.punctuality_medium, "b", linewidth=1.5, label="Regular")
        ax1.plot(self.var_punctuality, self.punctuality_high, "g", linewidth=1.5, label="Good")
        ax1.set_title("Punctuality")
        ax1.legend()

        ax2.plot(self.var_price, self.price_low, "g", linewidth=1.5, label="Low")
        ax2.plot(self.var_price, self.price_medium, "b", linewidth=1.5, label="Regular")
        ax2.plot(self.var_price, self.price_high, "r", linewidth=1.5, label="High")
        ax2.set_title("Price")
        ax2.legend()

        ax3.plot(self.var_supplier, self.supplier_wait, "r", linewidth=1.5, label="Wait")
        ax3.plot(self.var_supplier, self.supplier_implement, "g", linewidth=1.5, label="Implement")
        ax3.set_title("Supplier")
        ax3.legend()

        for ax in [ax0, ax1, ax2, ax3]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()

        plt.tight_layout()
        plt.show()
    
    def evaluate_supplier(self, supplier: Supplier, quotation_ecn: ECN):
        supplier_id = supplier.id
        df_prices = self.df[["Supplier ID", "Price"]].groupby("Supplier ID")["Price"].sum()
        
        crisp_delivery_time = misc.mean_not_outliers(self.df, "Delivery time", (self.df["Supplier ID"] == supplier_id) & (self.df["Awarded"] == True))

        crisp_price = df_prices.loc[supplier_id]

        crisp_punctuality = len(self.df[(self.df["Supplier ID"] == supplier_id) & (self.df["Awarded"] == True) & (self.df["OTD"] == True)]) / len(self.df[(self.df["Supplier ID"] == supplier_id) & (self.df["Awarded"] == True)])

        # Assign membership degree
        price_level_low = fuzzy.interp_membership(self.var_price, self.price_low, crisp_price)
        price_level_medium = fuzzy.interp_membership(self.var_price, self.price_medium, crisp_price)
        price_level_high = fuzzy.interp_membership(self.var_price, self.price_high, crisp_price)

        punctuality_level_low = fuzzy.interp_membership(self.var_punctuality, self.punctuality_low, crisp_punctuality)
        punctuality_level_medium = fuzzy.interp_membership(self.var_punctuality, self.punctuality_medium, crisp_punctuality)
        punctuality_level_high = fuzzy.interp_membership(self.var_punctuality, self.punctuality_high, crisp_punctuality)

        delivery_time_level_low = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_low, crisp_delivery_time)
        delivery_time_level_medium = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_medium, crisp_delivery_time)
        delivery_time_level_high = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_high, crisp_delivery_time)

        # Rule application
        rule_1 = min(delivery_time_level_low, price_level_low) # Implement

        rule_2 = min(delivery_time_level_low, price_level_medium, punctuality_level_high)  # Implement

        rule_3 = min(delivery_time_level_low, price_level_high)  # Wait

        rule_4 = min(delivery_time_level_medium, price_level_low, punctuality_level_high)  # Implement

        rule_5 = min(delivery_time_level_medium, price_level_medium, punctuality_level_low)    # Wait

        rule_6 = min(delivery_time_level_medium, price_level_medium, punctuality_level_high)   # Implement

        rule_7 = min(delivery_time_level_high, price_level_high)   # Wait

        rule_8 = min(delivery_time_level_high, price_level_medium, punctuality_level_low)  # Wait

        rule_9 = min(delivery_time_level_low, price_level_medium, punctuality_level_low)  # Wait

        rule_10 = min(delivery_time_level_low, price_level_medium, punctuality_level_medium)  # Implement

        wait_strength = max(rule_3, rule_5, rule_7, rule_8, rule_9)
        implement_strength = max(rule_1, rule_2, rule_4, rule_6, rule_10)

        supplier_activation_wait = np.fmin(wait_strength,self.supplier_wait)
        supplier_activation_implement = np.fmin(implement_strength, self.supplier_implement)

        aggregated = np.fmax.reduce([supplier_activation_wait, supplier_activation_implement])

        # Defuzzification
        supplier_score = fuzzy.defuzz(self.var_supplier, aggregated, "centroid")
        supplier_activation = fuzzy.interp_membership(self.var_supplier, aggregated, supplier_score)

        if supplier_activation_implement.max() > supplier_activation_wait.max():
          action = "Implement"
        else:
            action = "Wait"

        return {
        "Supplier ID": supplier_id,
        "Score": supplier_score,
        "Wait": supplier_activation_wait.max(),
        "Implement": supplier_activation_implement.max(),
        "Action": action
        }