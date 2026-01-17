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
ItemMaster
    Provides a Pandas' DataFrame with a standardized column set.
PartNumber
    Represents a single material or part number.
Project
    Represents a NPI (New Product Introduction) project.
Quotation
    Represents a supplier's quotation.
Supplier
    Represents a supplier.
"""

## Dependency modules
from datetime import date, timedelta    # To work dates.
from dotenv import load_dotenv  # variables.
from os import getenv
from statistics import mean, stdev
from random import choice, random       # To generate random numbers.
import numpy as np                      # To generate random numbers.
import pandas as pd                     # To work with data frames.

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


class ItemMaster:
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


class PartNumber:
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
        Initialize the PartNumber object.

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
    items : list[PartNumber]
        List of part numbers released in this ECN.
    quotations : list[Quotation]
        List of quotations related to this ECN.
    readiness : dict[str, bool]
        Dictionary containing bools (True or False) indicating id the ECN implementation met the dates of the project.
    status : str
        ECN status.

    Methods
    -------
    display_as_df():
        Returns a DataFrame containing the columns 'Project', 'ECN', 'ECN release', 'Part number', 'Complexity', 'EAU',
        and its values.
    reset():
        Returns ECN status to default: "Under review" and erases quotation list.
    """
    instances = 0

    def __init__(self, project: Project, ecn_date: date, pn_list: list[PartNumber]):
        """
        Initialize the ECN object.

        Parameters:
            project (Project): Project related to this ECN.
            ecn_date (date): Date when ECN was released in date format.
            pn_list (list[PartNumber]): List of part numbers released in this ECN.
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
        The supplier quote an ECN.

        Parameters:
            ecn (ECN): ECN to quote.
            rfq_date (date): Date when Sourcing team send RFQ (Request For Quotation).
            lead_time (int): Lead time for this specific quotation. It can be different to supplier's standard lead
                             time. It is 0 by default.

        Returns:
            DataFrame: Quotation in Pandas' DataFrame format.
        """
        not_quoted_yet = True   # Indicator of quotation status for the ECN. Its initial status is True.
        for quotation in self.quotations:
            if quotation.ecn.ecn_id == ecn.ecn_id:  # Check is the supplier already quoted this ECN, if it already quoted it
                print(f"{self.name} already quoted {ecn.ecn_id}.")  # Prints a warning if the supplier already quoted the ECN.
                not_quoted_yet = False  # If supplier already quoted the ECN changes status to False.

        if not_quoted_yet:  # In case the supplier did not quote it:
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
        List of inactive suppliers. RFQs will not be sent to these suppliers.
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

        self.item_master = ItemMaster().df

        self.part_kinds = {
            "A": {  # Distributor
                "average": 1.2222222,
                "stdev": 2.04524,
                "complexity": {"low": 0.6818181818182, "medium": 0.318181818181818, "high": 0},
                "parts": []
            },
            "B": {  # Liquid tube
                "average": 0.5,
                "stdev": 0.857493,
                "complexity": {"low": 1/3, "medium": 2/3, "high": 0},
                "parts": []
            },
            "C": {  # Equalizer
                "average": 0.333333,
                "stdev": 0.766965,
                "complexity": {"low": 1, "medium": 0, "high": 0},
                "parts": []
            },
            "D": {  # Suction tube
                "average": 0.611111,
                "stdev": 0.777544,
                "complexity": {"low": 0.090909090909090909, "medium": 0.727272727272727, "high": 0.181818181818182},
                "parts": []
            },
            "E": {  # Distributor
                "average": 0.6666667,
                "stdev": 1.940285,
                "complexity": {"low": 0, "medium": 0, "high": 1},
                "parts": []
            },
            "F": {  # Header
                "average": 0.055556,
                "stdev": 0.235702,
                "complexity": {"low": 0, "medium": 0, "high": 1},
                "parts": []
            },
            "G": {  # Valve assembly
                "average": 0.44444,
                "stdev": 0.783823,
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
        Create a new supplier. This method calls Supplier class.

        Parameters:
            name (str): Name of the supplier.
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
        Generate a specific quantity of ECNs for a specific Project. The quantity of part numbers for every ECN is
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
            ecn_eau = max(round(np.random.normal(loc=self.µ_eau_qty, scale=self.σ_eau_qty)), self.min_eau_qty)

            while len(ecn_part_numbers) == 0:   # This while ensures that the ECN has at least one part number.
                for key in self.part_kinds.keys():
                    kind_complexity_keys = list(self.part_kinds[key]["complexity"].keys())
                    kind_complexity_probabilities = list(self.part_kinds[key]["complexity"].values())

                    for j in range(max(int(np.random.normal(self.part_kinds[key]["average"],
                                                            self.part_kinds[key]["stdev"])), 0)):
                        category_part_number = len(self.part_kinds[key]["parts"]) + 1
                        complexity = np.random.choice(kind_complexity_keys, p=kind_complexity_probabilities)

                        part_number = PartNumber(pn=f"A0{key}{str(category_part_number).zfill(6)}",
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
        Search an ECN in the ecn list by ID and return it.

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

                part_number = PartNumber(pn=f"A0{pn[2]}{str(category_part_number).zfill(6)}", complexity=complexity,
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
        more information about the required columns, please take as reference the attribute 'columns' of ItemMaster.

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

            part_number = PartNumber(pn=pn, complexity=pn_complexity, eau=pn_eau)

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
        Implement all the part numbers in an ECN with a chosen supplier.

        Parameters:
            ecn (ECN): ECN to implement.
            awarded_supplier (Supplier): Supplier to implement the ECN with.
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
                    if quotation.awarded:
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
        µ_delivery_time, σ_delivery_time = (awarded_supplier.µ_delivery_time, awarded_supplier.σ_delivery_time)
        µ_documents_time, σ_documents_time = (awarded_supplier.µ_isir_documents_upload,
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
        Implement the ECNs with random supplier to generate an initial dataset for a specific project. Supplier
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
        Calculate the statistical parameters (mean and standard deviation) for sample delivery time of awarded part
        numbers.

        Returns:
            tuple: (mean, standard deviation)
        """
        punctuality_list = []

        for supplier_id in self.item_master[(self.item_master["Awarded"] == True)]["Supplier ID"].unique():
            punctuality = len(self.item_master[(self.item_master["Supplier ID"] == supplier_id) & (self.item_master["Awarded"] == True) & (self.item_master["OTD"] == True)]) / len(self.item_master[(self.item_master["Supplier ID"] == supplier_id) & (self.item_master["Awarded"] == True)])

            punctuality_list.append(punctuality)

        return mean(punctuality_list), stdev(punctuality_list)