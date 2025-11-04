###############################################################################
###                                                                         ###
###                         NPI Sourcing simulation                         ###
###                    Saúl R. Morales © 2025 MIT License                   ###
###                                                                         ###
###   This module provides the required clases to simulate the Sourcing     ###
###   NPI process. It is intended to simulate the process of sourcing       ###
###   copper piping in a new product introduction (NPI) project for HVAC    ###
###   industry, from Design Freeze (P2) to Start of Production (P4)         ###
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
This module provides the required clases to simulate the Sourcing NPI process. It is intended to simulate a basic
process of sourcing in a new product introduction (NPI) project.

Classes
-------
ECN
    Represents an ECN (Engineering Change Notification) as a set of part numbers.
Environment
    Represents the environment were the NPI project takes place and objects interact.
Fuzzy_Model
    Provides a fuzzy logic Mamdni's model to evaluate suppliers.
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

## Depency modules
from datetime import date, timedelta    # To work dates.
from logging import exception
from unittest import case

from dotenv import load_dotenv  # variables.
from math import ceil, floor
from os import getenv
from statistics import mean, stdev
from random import choice, random       # To generate random numbers.
import matplotlib.pyplot as plt
import colors
import misc
import numpy as np                      # To generate random numbers.
import pandas as pd                     # To work with data frames.
import skfuzzy as fuzzy

load_dotenv()   # Load .env file with Environment secrets.

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
            df_date (date): Date of design freeze. Design freeze is the date when non-critical changes to the project
                            cease. ECNs are released near this date.
            mcs_date (date): Date of MCS. Samples need to arrive before this date, but they do not need to be PPAP
                             approved.
            pilot_date (date): Date of pilot. Building pilot units. Material need to be PPAP approved and in SAP
                               contract before this date; procurement is made by regular purchasing process.
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
    Standardized column format for the data frame.

    Attributes
    ----------
    df : DataFrame
        Pandas' data frame.
    """

    def __init__(self):
        """Initialize the Item Master object."""
        columns = {
            # Simulation data
            "Iteration": [],
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
            eau (int): Estimated Annual Use, is the quantity of the material that is intended to consume during the
                       fiscal year.
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
        Returns a DataFrame containing the columns 'Project', 'ECN', 'ECN release', 'Part number', 'Complexity', 'EAU',
        and its values.
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
        self.status = "Under review"  # Status are: "Under review", "Engineering release" and "Release for production"
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

    def reset(self):
        self.status = "Under review"
        self.quotations = []

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
        Counter of created Suppliers.
    id : str
        Supplier SAP ID.
    name : str
        Supplier's name.
    quotations : list[Quotation]
        List of the quotations that the supplier had issued.
    awarded_ecns : list[ECN]
        List of awarded ECNs to the supplier.
    delivery_profile : str
        It determines the delivery speed.
    quotation_profile : str
        It determines the quotation time.
    price_profile : str
        It determines the prices.
    punctuality_profile : str
        It determines the punctuality of supplier for sample delivery.
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

    def __init__(
            self,
            name: str,
            delivery_profile: str = "regular",
            quotation_profile: str = "regular",
            price_profile: str = "regular",
            punctuality_profile: str = "regular",
            standard_lt: int = 0
    ):
        """
        Initializes the supplier object.

        Parameters:
            name (str): Supplier name.
            delivery_profile (str): It determines the delivery speed. Valid options are: 'low', 'regular', and 'high'.
                                    Default is 'regular'.
            quotation_profile (str): It determines the quotation time. Valid options are: 'low', 'regular', and 'high'.
                                     Default is 'regular'.
            price_profile (str): It determines the prices. Valid options are: 'low', 'regular', and 'high'. Default is
                                 'regular'.
            punctuality_profile (str): It determines the punctuality of supplier for sample delivery. Valid options are:
                                       'low', 'regular', and 'high'. Default is 'regular'.
            standard_lt (int): This is the lead time in days that the supplier usually writes in their quotation,
                               however in real life this time usually is not met. It is 0 by default.
        """
        Supplier.instances += 1
        self.id = f"1{str(Supplier.instances).zfill(7)}"    # Supplier ID is automatically calculated according to the
                                                            # current quantity of created suppliers.
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

        self.eta_difference = { # First number is the average, second number is the standard deviation.
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
        """Supplier name when the Supplier is printed."""
        return self.name

    def quote(self, ecn: ECN, rfq_date: date, lead_time: int = 0):
        """
        The supplier quote a ECN.

        Parameters:
            ecn (ECN): ECN to quote.
            rfq_date (date): Date when Sourcing team send RFQ (Request For Quotation).
            lead time (int): Lead time for this specific quotation. It can be different to supplier's standard lead
                             time. It is 0 by default.

        Returns:
            DataFrame: Quotation in Pandas' DataFrame format.
        """
        not_quoted_yet = True   # Indicator of quotation status for the ECN. Its initial status is True.
        for quotation in self.quotations:
            if quotation.ecn.ecn_id == ecn.ecn_id:  # Check is the supplier already quoted this ECN, if it already quoted it
                print(f"{self.name} already quoted {ecn.ecn_id}.")  # Prints a warning if the supplier already quoted the ECN.
                not_quoted_yet = False  # If supplier already quoted the ECN changes status to False.

        if not_quoted_yet:  # In case the supplier did not quoted it:
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

                quotation.df.loc[len(quotation.df)] = [ # Adds the part number data to the quotation DataFrame.
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

            self.quotations.append(quotation)   # Append the quotation's DataFrame to quotation's list.
            return quotation.df


class Environment:
    """
    Represents the environment were the NPI project takes place and objects interact.

    Attributes
    ----------
    suppliers : list[Supplier]
        List of all the created suppliers in the environment.
    active_suppliers : list[Supplier]
        List of active suppliers for quotation.
    inactive_suppliers : list[Supplier]
        List of inactive suppliers. RFQs will not be send to these suppliers.
    ecns : list[ECN]
        List of created ECNs.
    item_master : DataFrame
        Master DataFrame of created PartNumber. This record all the information related to a part number and their
        interaction with the suppliers.
    part_kinds : dict[str: dict [str: float | dict[str: float] | list[PartNumber]]]
        There are seven kind of copper tubing; this dictionary contains the mean, standard deviation, price profile
        parameters according the technical complexity, and a list of created part numbers for each kind.
    environment_times : dict[str: int | tuple[float]]
        These dictionary provides the statistical parameters (mean and standard deviation) for each activity in the NPI
        Sourcing process.
    µ_eau_qty : float
        Average estimated anual usage (EAU) for the part numbers in the ECN for a low-volume plant.
    σ_eau_qty : float
        Standard deviation for EAU for the part numbers in the ECN for a low-volume plant.
    min_eau_qty : int
        Minimum EAU quantity.
    """
    def __init__(self):
        """Initializes the Environment object."""
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
        """
        Add a supplier to the Environment.

        Parameters:
            supplier (Supplier): Supplier to add.
        """
        self.suppliers.append(supplier) # Append Supplier to Environment's supplier list.
        self.active_suppliers.append(supplier)  # Activate supplier for RFQs.

    def add_suppliers(self, suppliers: list[Supplier]):
        """
        Add the suppliers in a list to the Environment.

        Parameters:
            suppliers (list[Supplier]): List of suppliers to add.
        """
        for supplier in suppliers:
            self.add_supplier(supplier)

    def activate_supplier(self, supplier: Supplier):
        """
        Enable a Supplier to quote ECNs.

        Parameters:
            supplier (Supplier): Supplier to activate.
        """
        if supplier in self.active_suppliers:   # Verify if supplier is already active.
            print(f"{supplier.name} is already active.")    # If they are active, print a warning.
        else:
            self.active_suppliers.append(supplier)  # Add Supplier to active_suppliers list.
            self.inactive_suppliers.remove(supplier)    # Remove Supplier from inactive_Suppliers list.

    def create_supplier(self, name: str, delivery_profile: str = "regular", quotation_profile: str = "regular", price_profile: str = "regular", punctuality_profile: str = "regular", active: bool = True):
        """
        Create a new supplier. This method calls Supplier object.

        Parameters:
            delivery_profile (str): It determines the delivery speed. Valid options are: 'low', 'regular', and 'high'.
                                    Default is 'regular'.
            quotation_profile (str): It determines the quotation time. Valid options are: 'low', 'regular', and 'high'.
                                     Default is 'regular'.
            price_profile (str): It determines the prices. Valid options are: 'low', 'regular', and 'high'. Default is
                                 'regular'.
            punctuality_profile (str): It determines the punctuality of supplier for sample delivery. Valid options are:
                                       'low', 'regular', and 'high'. Default is 'regular'.
            active (bool): True to create an active Supplier; False to create an inactive Supplier.
        """
        for supplier in self.suppliers:
            if supplier.name == name:                                                           # Print a warning if
                raise Exception(f"Supplier {name} already exists. Its ID is {supplier.id}.")    # there is already a
                                                                                                # supplier with the same
                                                                                                # number.

        supplier = Supplier(name, delivery_profile, quotation_profile, price_profile, punctuality_profile)
        self.add_supplier(supplier)

        if not active:
            self.deactivate_supplier(supplier)

    def deactivate_supplier(self, supplier: Supplier):
        """
        Disables a Supplier to quote ECNs.

        Parameters:
            supplier (Supplier): Supplier to deactivate.
        """
        self.inactive_suppliers.append(supplier)
        self.active_suppliers.remove(supplier)

    def gen_ecns(self, project: Project, qty: int):
        """
        Generate a specific quantity of ECNs for an specific Project. The quantity of part numbers for every ECN is
        generated randomly.

        Parameters:
            project (Project): Project to generate the ECNs.
            qty (int): Quantity of ECNs to generate.

        Returns:
            list: A list containing the ECNs for the Project.
        """
        ecns = []   # List to store the created ECNs.

        for i in range(qty):
            ecn_part_numbers = []   # List to store the created part numbers of the ECN,
            ecn_eau = max(round(np.random.normal(loc=self.μ_eau_qty, scale=self.σ_eau_qty)), self.min_eau_qty)

            while len(ecn_part_numbers) == 0:   # This while ensures that the ECN has at least one part number.
                for key in self.part_kinds.keys():
                    kind_complexity_keys = list(self.part_kinds[key]["complexity"].keys())
                    kind_complexity_probabilities = list(self.part_kinds[key]["complexity"].values())

                    for j in range(max(int(np.random.normal(self.part_kinds[key]["average"],
                                                            self.part_kinds[key]["stdev"])), 0)):
                        category_part_number = len(self.part_kinds[key]["parts"]) + 1
                        complexity = np.random.choice(kind_complexity_keys, p=kind_complexity_probabilities)

                        part_number = Part_Number(pn=f"A0{key}{str(category_part_number).zfill(6)}",
                                                  complexity=complexity, eau=ecn_eau)

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
        """
        Search a ECN in the ecn list by ID and return it.

        Parameters:
            search_reference (str): ECN ID to search.
        
        Returns:
            ECN: ECN object.
        """
        for ecn in self.ecns:
            if ecn.ecn_id == search_reference:
                return ecn

    def get_supplier(self, search_mode: str, reference: str):
        """
        Search a supplier in the supplier list by a reference and return them.

        Parameters:
            search_mode (str): Search term. Valid options are 'name' and 'id'. Use 'name to search a supplier by their
                               name; use 'id' to search them by their ID.
            reference (str): Supplier name or ID.
        
        Returns:
            Supplier: Supplier object.
        """
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

    def import_ecns_from_df(self, project: Project, df: pd.DataFrame):
        """
        Import ECNs and part numbers from a DataFrame. This method is useful when you want to recreate a simulation
        using the same part numbers as in a past session. The DataFrame must contain these columns: 'Project', 'ECN',
        ECN release', 'Part number', 'Complexity', and 'EAU'.

        Parameters:
            project (Project): Object representing a project to store this ECNs.
            df (DataFrame): DataFrame containing the information regarding the ECNs and part numbers to import.

        Returns:
            list: List containing the ECNs.
        """
        ecns = []
        for ecn in df["ECN"].unique():
            ecn_part_numbers = []
            ecn_date = df[df["ECN"] == ecn]["ECN release"].iloc[0].date()

            for pn in df[df["ECN"] == ecn]["Part number"]:
                category_part_number = len(self.part_kinds[pn[2]]["parts"]) + 1
                complexity = df[df["Part number"] == pn]["Complexity"].item()
                eau = df[df["Part number"] == pn]["EAU"].item()

                part_number = Part_Number(pn=f"A0{pn[2]}{str(category_part_number).zfill(6)}", complexity=complexity,
                                          eau=eau)

                self.part_kinds[pn[2]]["parts"].append(part_number)
                ecn_part_numbers.append(part_number)

            ecn = ECN(project=project, ecn_date=ecn_date, pn_list=ecn_part_numbers)
            self.ecns.append(ecn)
            project.ecns.append(ecn)
            ecns.append(ecn)

        return ecns

    def import_training_df(self, training_df: pd.DataFrame):
        """
        Import a dataset containing the historical information or a training dataset generated by the simulation. For
        more information about the required columns, please take as reference the attribute 'columns' of Item_Master.

        Parameters:
            training_df (DataFrame): DataFrame containing the historical o training dataset.
        """
        self.item_master = training_df

        project_names = training_df["Project"].unique()
        project_data = {}
        for project_name in project_names:
            project_data[project_name] = Project(
                project_name,
                date(2025,9,18),
                date(2025,9,18),
                date(2025,9,18),
                date(2025,9,18)
            )

        ecn_list = training_df["ECN"].unique()
        ecn_part_list = {}
        for ecn_id in ecn_list:
            ecn_part_list[ecn_id] = []

        pn_list = training_df["Part number"].unique()
        for pn in pn_list:
            pn_ecn = training_df[training_df["Part number"] == pn]["ECN"].iloc[0]
            pn_complexity = training_df[training_df["Part number"] == pn]["Complexity"].iloc[0]
            pn_eau = training_df[training_df["Part number"] == pn]["EAU"].max()

            part_number = Part_Number(pn=pn, complexity=pn_complexity, eau=pn_eau)

            self.part_kinds[pn[2]]["parts"].append(part_number)
            ecn_part_list[pn_ecn].append(part_number)

        for ecn_id in ecn_part_list.keys():
            ecn_project = project_data[training_df[training_df["ECN"] == ecn_id]["Project"].iloc[0]]
            ecn = ECN(
                project=ecn_project,
                ecn_date=training_df[training_df["ECN"] == ecn_id]["ECN release"].max(),
                pn_list=ecn_part_list[ecn_id]
            )

            self.ecns.append(ecn)
            ecn_project.ecns.append(ecn)

    def quote_all_ecn_project_all_suppliers(self, project: Project):
        """
        Send a RFQ for all the ECNs of a specific project to all the active suppliers.

        Parameters:
            project (Project): Project to quote.
        """
        for ecn in project.ecns:
            self.quote_ecn_all_suppliers(ecn)

    def quote_ecn_all_suppliers(self, ecn: ECN):
        """
        Send a RFQ for a specific ECN to all the active suppliers.
        """
        µ_rfq_time, σ_rfq_time = self.environment_times["send_rfq"]
        ecn.status = "Engineering release"

        for supplier in self.active_suppliers:
            rfq_date = ecn.ecn_date + timedelta(days=max(round(np.random.normal(loc=µ_rfq_time, scale=σ_rfq_time)), 0))

            quotation = supplier.quote(ecn, rfq_date)
            self.item_master = pd.concat([self.item_master, quotation], ignore_index=True)

        return self.item_master[self.item_master["ECN"] == ecn.ecn_id]

    def quote_ecn_some_suppliers(self, ecn: ECN, quoting_suppliers: list[Supplier]):
        """
        Send RFQ for a specific ECN to all the suppliers in a list.

        Parameters:
            ecn (ECN): ECN to quote.
            quoting_suppliers (list[Supplier]): List containing the suppliers to send RFQ.
        """
        µ_rfq_time, σ_rfq_time = self.environment_times["send_rfq"]
        ecn.status = "Engineering release"

        for supplier in quoting_suppliers:
            if supplier in self.active_suppliers:
                rfq_date = ecn.ecn_date + timedelta(days=max(round(np.random.normal(loc=µ_rfq_time, scale=σ_rfq_time)),
                                                             0))

                quotation = supplier.quote(ecn, rfq_date)
                self.item_master = pd.concat([self.item_master, quotation], ignore_index=True)

                return self.item_master[self.item_master["ECN"] == ecn.ecn_id]
            else:
                print(f"{supplier.name} is inactive.")

    def implement_ecn(self, ecn: ECN, awarded_supplier: Supplier, overwrite: bool = False):
        """
        Implement all the part numbers in a ECN with a chosen supplier.

        Parameters:
            ecn (ECN): ECN to implement.
            awarded_supplier (Supplier): Supplie to implement the ECN with.
            overwrite (bool): If True, the ECN will be implemented overwriting the existing values for the attributes in
                              the related objects; this is useful when simulating the implementation with two or more
                              iterations. It is False by default.

        Returns:
            DataFrame: DataFrame containing the information of implement ECN.
        """
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
        µ_documents_time, σ_documents_time = (awarded_supplier.μ_isir_documents_upload,
                                              awarded_supplier.σ_isir_documents_upload)
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
        ecn.status = "Released for production"
        return self.item_master[(self.item_master["ECN"] == ecn.ecn_id) & (self.item_master["Supplier ID"] == awarded_supplier.id)]

    def quote_all_ecns(self):
        """
        Send RFQ for all the ECNs of all the projects to all the active suppliers.

        Returns:
            DataFrame: DataFrame containing the quoted ECNs.
        """
        for ecn in self.ecns:
            self.quote_ecn_all_suppliers(ecn)
        return self.item_master

    def gen_initial_item_master_df(self):
        """
        Implement the ECNs with random supplier to generate an initial dataset. Supplier selection is done using a
        uniform random distribution.

        Returns:
            DataFrame: Item master.
        """
        for ecn in self.ecns:
            random_supplier = choice(self.suppliers)

            self.implement_ecn(ecn, random_supplier)
        return self.item_master

    def gen_initial_item_master_df_project(self, project: Project):
        """
        Implement the ECNs with random supplier to generate an initial dataset for an specific project. Supplier
        selection is done using a uniform random distribution.

        Returns:
            DataFrame: Item master.
        """
        for ecn in project.ecns:
            random_supplier = choice(self.suppliers)

            self.implement_ecn(ecn, random_supplier)

        return self.item_master

    def get_µσ_punctuality(self):
        """
        Calculate the statististical parameters (mean and standard deviation) for sample delivery time of awarded part
        numbers.

        Returns:
            tupple: (mean, standard deviation)
        """
        punctuality_list = []

        for supplier_id in self.item_master[(self.item_master["Awarded"] == True)]["Supplier ID"].unique():
            punctuality = len(self.item_master[(self.item_master["Supplier ID"] == supplier_id) & (self.item_master["Awarded"] == True) & (self.item_master["OTD"] == True)]) / len(self.item_master[(self.item_master["Supplier ID"] == supplier_id) & (self.item_master["Awarded"] == True)])

            punctuality_list.append(punctuality)

        return (mean(punctuality_list), stdev(punctuality_list))


class Fuzzy_Model:
    """
    Fuzzy logic model to evaluate suppliers on NPI projects.
    """
    def __init__(
            self,
            df_item_master: pd.DataFrame,
            ref_supplier: Supplier,
            quotation_ecn: ECN,
            evaluation_priority: str,   # Options are: "time" and "spend"
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
                max_delivery_time = ceil(max(avg_delivery_time + 3 * std_delivery_time,self.df[(self.df["Awarded"] == True)]["Delivery time"].max()))
                quotations_df = self.df[self.df["Part number"].isin(quoted_pn)]
                self.spend_df = quotations_df[["Supplier ID", "FY Spend"]].groupby("Supplier ID").sum()["FY Spend"]

            case "spend":   # Revisar la obtención del spend, ya que para el objetivo 'spend' debe considerar únicamente lo del proyecto.
                self.spend_df = self.df[["Supplier ID", "FY Spend"]].groupby("Supplier ID").sum()["FY Spend"]
                max_delivery_time = ceil(self.df[(self.df["Awarded"] == True)]["Delivery time"].max())

            case _:
                exception("Evaluation priority must be 'time' or 'spend'.")

        avg_spend = (self.spend_df.mean()) / 100    # Igual en ambos casos
        std_spend = (self.spend_df.std()) / 100 # Igual en ambos casos
        min_spend = floor((self.spend_df.min()) / 100) # Igual en ambos casos

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

        self.var_due_time = np.arange(0,721, 1)
        self.var_delivery_time = np.arange(0, max_delivery_time + 1, 1)
        self.var_spend = np.arange(0, max_spend + 1, 0.01)
        self.var_punctuality = np.arange(0, 2, 0.01)
        self.var_supplier = np.arange(0, 11, 0.01)

        self.due_time_low = fuzzy.trapmf(self.var_due_time, [0,0,30,60])
        self.due_time_medium = fuzzy.trimf(self.var_due_time, [30, 60, 90])
        self.due_time_high = fuzzy.trapmf(self.var_due_time, [60,90,720,720])

        self.delivery_time_low = fuzzy.trapmf(self.var_delivery_time, [0, 0, avg_delivery_time - std_delivery_time, avg_delivery_time])
        self.delivery_time_medium = fuzzy.trimf(self.var_delivery_time, [avg_delivery_time - std_delivery_time, avg_delivery_time, avg_delivery_time + std_delivery_time])
        self.delivery_time_high = fuzzy.trapmf(self.var_delivery_time, [avg_delivery_time, avg_delivery_time + std_delivery_time, max_delivery_time, max_delivery_time])

        if not new_supplier:
            self.punctuality_low = fuzzy.trapmf(self.var_punctuality, [0, 0, 0.25, 0.5])
            self.punctuality_medium = fuzzy.trimf(self.var_punctuality, [0.25, 0.5, 0.75])
            self.punctuality_high = fuzzy.trapmf(self.var_punctuality, [0.5, 0.75, 1, 1])

        self.spend_low = fuzzy.trapmf(self.var_spend, [0, min_spend, max(min_spend, avg_spend - std_spend), max(avg_spend - std_spend, avg_spend)])
        self.spend_medium = fuzzy.trimf(self.var_spend, [avg_spend - std_spend, avg_spend, avg_spend + std_spend])
        self.spend_high = fuzzy.trapmf(self.var_spend, [avg_spend, avg_spend + std_spend, max_spend, max_spend])

        match evaluation_priority:
            case "time":
                self.supplier_wait = fuzzy.trapmf(self.var_supplier, [0, 0, 5, 7.5])
                self.supplier_implement = fuzzy.trapmf(self.var_supplier, [2.5, 5, 10, 10])

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
                self.supplier_low = fuzzy.trimf(self.var_supplier, [0, 2.5, 5])
                self.supplier_medium = fuzzy.trimf(self.var_supplier, [2.5, 5, 7.5])
                self.supplier_high = fuzzy.trimf(self.var_supplier, [5, 7.5, 10])

            case _:
                exception("Evaluation priority must be 'time' or 'spend'.")

    def get_stats(self):
       return self.stats

    def plot_model(self):
        if not self.new_supplier or self.evaluation_priority == "spend":
            fig, [ax0, ax1, ax2, ax3, ax4] = plt.subplots(nrows=5, figsize=(8, 9))

            ax0.plot(self.var_due_time, self.due_time_low, "r", linewidth=1.5, label="Close")
            ax0.plot(self.var_due_time, self.due_time_medium, "b", linewidth=1.5, label="Near")
            ax0.plot(self.var_due_time, self.due_time_high, "g", linewidth=1.5, label="Far")
            ax0.set_title("Due time")
            ax0.legend()

            ax1.plot(self.var_delivery_time, self.delivery_time_low, "g", linewidth=1.5, label="Good")
            ax1.plot(self.var_delivery_time, self.delivery_time_medium, "b", linewidth=1.5, label="Regular")
            ax1.plot(self.var_delivery_time, self.delivery_time_high, "r", linewidth=1.5, label="Bad")
            ax1.set_title("Delivery time")
            ax1.legend()

            ax2.plot(self.var_spend, self.spend_low, "g", linewidth=1.5, label="Low")
            ax2.plot(self.var_spend, self.spend_medium, "b", linewidth=1.5, label="Regular")
            ax2.plot(self.var_spend, self.spend_high, "r", linewidth=1.5, label="High")
            ax2.set_title("FY Spend")
            ax2.legend()

            ax3.plot(self.var_punctuality, self.punctuality_low, "r", linewidth=1.5, label="Bad")
            ax3.plot(self.var_punctuality, self.punctuality_medium, "b", linewidth=1.5, label="Regular")
            ax3.plot(self.var_punctuality, self.punctuality_high, "g", linewidth=1.5, label="Good")
            ax3.set_title("Punctuality")
            ax3.legend()

            ax4.plot(self.var_supplier, self.supplier_wait, "r", linewidth=1.5, label="Wait")
            ax4.plot(self.var_supplier, self.supplier_implement, "g", linewidth=1.5, label="Implement")
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

            ax0.plot(self.var_due_time, self.due_time_low, "r", linewidth=1.5, label="Close")
            ax0.plot(self.var_due_time, self.due_time_medium, "b", linewidth=1.5, label="Near")
            ax0.plot(self.var_due_time, self.due_time_high, "g", linewidth=1.5, label="Far")
            ax0.set_title("Due time")
            ax0.legend()

            ax1.plot(self.var_delivery_time, self.delivery_time_low, "g", linewidth=1.5, label="Good")
            ax1.plot(self.var_delivery_time, self.delivery_time_medium, "b", linewidth=1.5, label="Regular")
            ax1.plot(self.var_delivery_time, self.delivery_time_high, "r", linewidth=1.5, label="Bad")
            ax1.set_title("Delivery time")
            ax1.legend()

            ax2.plot(self.var_spend, self.spend_low, "g", linewidth=1.5, label="Low")
            ax2.plot(self.var_spend, self.spend_medium, "b", linewidth=1.5, label="Regular")
            ax2.plot(self.var_spend, self.spend_high, "r", linewidth=1.5, label="High")
            ax2.set_title("FY Spend")
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

    def _evaluate_supplier_time_priority(self, quotation_ecn: ECN):
        sop_date = quotation_ecn.project.important_dates["SOP"]
        quotation_date = self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["ECN"] == quotation_ecn.ecn_id)]["Quotation date"].max()
        due_time = sop_date - quotation_date
        crisp_due_time = max(due_time.days, 0)

        due_time_level_low = fuzzy.interp_membership(self.var_due_time, self.due_time_low, crisp_due_time)
        due_time_level_medium = fuzzy.interp_membership(self.var_due_time, self.due_time_medium, crisp_due_time)
        due_time_level_high = fuzzy.interp_membership(self.var_due_time, self.due_time_high, crisp_due_time)
        
        # Assign membership degree
        crisp_spend = (self.spend_df.loc[self.evaluating_supplier_id]) / 100

        spend_level_low = fuzzy.interp_membership(self.var_spend, self.spend_low, crisp_spend)
        spend_level_medium = fuzzy.interp_membership(self.var_spend, self.spend_medium, crisp_spend)
        spend_level_high = fuzzy.interp_membership(self.var_spend, self.spend_high, crisp_spend)

        if self.new_supplier:
            crisp_delivery_time = self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["ECN"] == quotation_ecn.ecn_id)]["Lead time"].max()
        else:
            crisp_punctuality = len(self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True) & (self.df["OTD"] == True)]) / len(self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True)])

            punctuality_level_low = fuzzy.interp_membership(self.var_punctuality, self.punctuality_low, crisp_punctuality)
            punctuality_level_medium = fuzzy.interp_membership(self.var_punctuality, self.punctuality_medium, crisp_punctuality)
            punctuality_level_high = fuzzy.interp_membership(self.var_punctuality, self.punctuality_high, crisp_punctuality)

            crisp_delivery_time = self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["ECN"] == quotation_ecn.ecn_id)]["Lead time"].max()

        delivery_time_level_low = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_low, crisp_delivery_time)
        delivery_time_level_medium = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_medium, crisp_delivery_time)
        delivery_time_level_high = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_high, crisp_delivery_time)

        # Rule application
        if self.new_supplier:
            rule_1 = delivery_time_level_high # Wait

            rule_2 = min(due_time_level_low, max(delivery_time_level_low, delivery_time_level_medium), spend_level_high) # Wait

            rule_3 = min(max(due_time_level_low, due_time_level_medium), max(delivery_time_level_low, delivery_time_level_medium), max(spend_level_low, spend_level_medium))  # Implement

            rule_4 = min(due_time_level_medium, spend_level_high)  # Wait

            rule_5 = min(due_time_level_high, max(delivery_time_level_low, delivery_time_level_medium), spend_level_low)  # Implement

            rule_6 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high), max(spend_level_medium, spend_level_high)) # Wait

            wait_strength = max(rule_1, rule_2, rule_4, rule_6)
            implement_strength = max(rule_3, rule_5)

        else:
            rule_1 = min(due_time_level_low, delivery_time_level_low, punctuality_level_low) # Wait

            rule_2 = min(due_time_level_low, delivery_time_level_low, punctuality_level_medium, spend_level_high)  # Wait

            rule_3 = min(due_time_level_low, max(delivery_time_level_medium, delivery_time_level_high), max(punctuality_level_low, punctuality_level_medium))  # Wait

            rule_4 = min(due_time_level_low, delivery_time_level_medium, punctuality_level_high)  # Implement

            rule_5 = min(due_time_level_low, delivery_time_level_low, punctuality_level_medium, max(spend_level_low, spend_level_medium))  # Implement

            rule_6 = min(due_time_level_low, delivery_time_level_low, punctuality_level_high)    # Implement

            rule_7 = min(due_time_level_medium, max(delivery_time_level_low, delivery_time_level_medium), punctuality_level_low)   # Wait

            rule_8 = min(due_time_level_medium, max(delivery_time_level_low, delivery_time_level_medium), max(punctuality_level_medium, punctuality_level_high), max(spend_level_low, spend_level_medium))   # Implement

            rule_9 = min(due_time_level_medium, max(delivery_time_level_low, delivery_time_level_medium), max(punctuality_level_medium, punctuality_level_high), spend_level_high)  # Wait

            rule_10 = min(due_time_level_medium, delivery_time_level_high)  # Wait

            rule_11 = min(due_time_level_high, delivery_time_level_low, spend_level_low)  # Implement

            rule_12 = min(due_time_level_high, delivery_time_level_low, punctuality_level_high, spend_level_medium)  # Implement

            rule_13 = min(due_time_level_high, delivery_time_level_low, max(punctuality_level_low, punctuality_level_medium), max(spend_level_medium, spend_level_high))  # Wait

            rule_14 = min(due_time_level_high, delivery_time_level_low, punctuality_level_high, spend_level_high)  # Wait

            rule_15 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high), punctuality_level_low, spend_level_medium)  # Wait

            rule_16 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high), spend_level_high)  # Wait

            rule_17 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high), punctuality_level_low, spend_level_low)  # Implement

            rule_18 = min(due_time_level_high, max(delivery_time_level_medium, delivery_time_level_high), max(punctuality_level_medium, punctuality_level_high), max(spend_level_low, spend_level_medium))  # Implement

            wait_strength = max(rule_1, rule_2, rule_3, rule_7, rule_9, rule_10, rule_13, rule_14, rule_15, rule_16)
            implement_strength = max(rule_4, rule_5, rule_6, rule_8, rule_11, rule_12, rule_17, rule_18)

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
            for i in range(7,19):
                stats[f"Rule {i}"] = np.nan
        else:
            rule_number = 1
            for rule in [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10, rule_11, rule_12, rule_13, rule_14, rule_15, rule_16, rule_17, rule_18]:
                stats[f"Rule {rule_number}"] = rule
                rule_number += 1

        return stats

    def _evaluate_supplier_spend_priority(self, project: Project):
        crisp_price = self.df[(self.df["Project"] == project.name) & (self.df["Supplier ID"] == self.evaluating_supplier_id)]["FY Spend"].sum() / 100

        if self.new_supplier:
            crisp_delivery_time = self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id)]["Lead time"].mean()

        else:
            crisp_punctuality = len(self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True) & (self.df["OTD"] == True)]) / len(self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True)])
            crisp_delivery_time = self.df[(self.df["Supplier ID"] == self.evaluating_supplier_id) & (self.df["Awarded"] == True)]["Delivery time"].mean()

            punctuality_level_low = fuzzy.interp_membership(self.var_punctuality, self.punctuality_low, crisp_punctuality)
            punctuality_level_medium = fuzzy.interp_membership(self.var_punctuality, self.punctuality_medium, crisp_punctuality)
            punctuality_level_high = fuzzy.interp_membership(self.var_punctuality, self.punctuality_high, crisp_punctuality)

        # Assign membership degree
        price_level_low = fuzzy.interp_membership(self.var_spend, self.spend_low, crisp_price)
        price_level_medium = fuzzy.interp_membership(self.var_spend, self.spend_medium, crisp_price)
        price_level_high = fuzzy.interp_membership(self.var_spend, self.spend_high, crisp_price)

        delivery_time_level_low = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_low, crisp_delivery_time)
        delivery_time_level_medium = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_medium, crisp_delivery_time)
        delivery_time_level_high = fuzzy.interp_membership(self.var_delivery_time, self.delivery_time_high, crisp_delivery_time)

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
            rule_1 = min(delivery_time_level_low, max(price_level_low, price_level_medium), punctuality_level_low)  # Regular

            rule_2 = min(delivery_time_level_low, max(price_level_low, price_level_medium), max(punctuality_level_medium, punctuality_level_high))  # High

            rule_3 = min(delivery_time_level_low, price_level_high, punctuality_level_low)  # Low

            rule_4 = min(delivery_time_level_low, price_level_high, max(punctuality_level_medium, punctuality_level_high))  # Regular

            rule_5 = min(delivery_time_level_medium, price_level_low, punctuality_level_low)  # Regular

            rule_6 = min(max(delivery_time_level_medium, delivery_time_level_high), max(price_level_low, price_level_medium), punctuality_level_medium)  # Regular

            rule_7 = min(max(delivery_time_level_medium, delivery_time_level_high), max(price_level_low, price_level_medium), punctuality_level_high)  # High

            rule_8 = min(delivery_time_level_medium, max(price_level_medium, price_level_high), punctuality_level_low)  # Low

            rule_9 = min(max(delivery_time_level_medium, delivery_time_level_high), price_level_high, punctuality_level_medium)  # Low

            rule_10 = min(max(delivery_time_level_medium, delivery_time_level_high), price_level_high, punctuality_level_high)  # Medium

            rule_11 = min(delivery_time_level_high, punctuality_level_low)  # Low

            low_strength = max(rule_3, rule_8, rule_9, rule_11)
            medium_strength = max(rule_1, rule_4, rule_5, rule_6, rule_10)
            high_strength = max(rule_2, rule_7)

        supplier_activation_low = np.fmin(low_strength, self.supplier_low)
        supplier_activation_medium = np.fmin(medium_strength, self.supplier_medium)
        supplier_activation_high = np.fmin(high_strength, self.supplier_high)

        supplier_0 = np.zeros_like(self.var_supplier)

        aggregated = np.fmax.reduce([supplier_activation_low, supplier_activation_medium, supplier_activation_high])

        # Defuzzification
        supplier_score = fuzzy.defuzz(self.var_supplier, aggregated, "centroid")
        supplier_activation = fuzzy.interp_membership(self.var_supplier, aggregated, supplier_score)

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
            for i in range(5,12):
                stats[f"Rule {i}"] = np.nan
        else:
            rule_number = 1
            for rule in [rule_1, rule_2, rule_3, rule_4, rule_5, rule_6, rule_7, rule_8, rule_9, rule_10, rule_11]:
                stats[f"Rule {rule_number}"] = rule
                rule_number += 1

        return stats