import pandas as pd
import os

def read_seasons(seasons_path):
    try:
        with open(seasons_path, 'r') as f:
            seasons = f.read()
    except FileNotFoundError:
        print(f'{seasons_path} does not exist' )
        raise 

    return seasons.split('\n')

def school_name_transform(school_name):
    try:

        school_name = school_name.lower()
        school_name = school_name.replace(" & ", " ")
        school_name = school_name.replace("&", "")
        school_name = school_name.replace("ncaa", "")
        school_name = school_name.strip()
        school_name = school_name.replace(" ", "-")
        school_name = school_name.replace("(", "")
        school_name = school_name.replace(")", "")
        school_name = school_name.replace(".", "")
        school_name = school_name.replace("'", "")

        if school_name == "siu-edwardsville":
            school_name = "southern-illinois-edwardsville"
        elif school_name == "vmi":
            school_name = "virginia-military-institute"
        elif school_name == "uc-davis":
            school_name = "california-davis"
        elif school_name == "uc-irvine":
            school_name = "california-irvine"
        elif school_name == "uc-riverside":
            school_name = "california-riverside"
        elif school_name == "uc-santa-barbara":
            school_name = "california-santa-barbara"
        elif school_name == "uc-san-diego":
            school_name = "california-san-diego"
        elif school_name == "university-of-california":
            school_name = "california"
        elif school_name == "cal-state-long-beach":
            school_name = "long-beach-state"
        elif school_name == "louisiana":
            school_name = "louisiana-lafayette"
        elif school_name == "texas-rio-grande-valley":
            school_name = "texas-pan-american"
        elif school_name == "byu":
            school_name = "brigham-young"
        elif school_name == "etsu":
            school_name = "east-tennessee-state"
        elif school_name == "liu":
            school_name = "long-island-university"
        elif school_name == "lsu":
            school_name = "louisiana-state"
        elif school_name == "nc-state":
            school_name = "north-carolina-state"
        elif school_name == "ole-miss":
            school_name = "mississippi"
        elif school_name == "pitt":
            school_name = "pittsburgh"
        elif school_name == "penn":
            school_name = "pennsylvania"
        elif school_name == "saint-marys":
            school_name = "saint-marys-ca"
        elif school_name == "smu":
            school_name = "southern-methodist"
        elif school_name == "tcu":
            school_name = "texas-christian"
        elif school_name == "umbc":
            school_name = "maryland-baltimore-county"  
        elif school_name == "umass":
            school_name = "massachusetts"    
        elif school_name == "umass-lowell":
            school_name = "massachusetts-lowell"  
        elif school_name == "unc":
            school_name = "north-carolina"  
        elif school_name == "unc-asheville":
            school_name = "north-carolina-asheville" 
        elif school_name == "unc-greensboro":
            school_name = "north-carolina-greensboro" 
        elif school_name == "unc-wilmington":
            school_name = "north-carolina-wilmington" 
        elif school_name == "ucf":
            school_name = "central-florida" 
        elif school_name == "uab":
            school_name = "alabama-birmingham"
        elif school_name == "the-citadel":
            school_name = "citadel"
        elif school_name == "ucsb":
            school_name = "california-santa-barbara"
        elif school_name == "uconn":
            school_name = "connecticut"
        elif school_name == "umkc":
            school_name = "missouri-kansas-city"
        elif school_name == "unlv":
            school_name = "nevada-las-vegas"
        elif school_name == "utep":
            school_name = "texas-el-paso"
        elif school_name == "utsa":
            school_name = "texas-san-antonio"
        elif school_name == "ut-arlington":
            school_name = "texas-arlington"
        elif school_name == "usc":
            school_name = "nsouthern-california"
        elif school_name == "vcu":
            school_name = "virginia-commonwealth"
        elif school_name == "uic":
            school_name = "illinois-chicago"
        elif school_name == "little-rock":
            school_name = "arkansas-little-rock"
        elif school_name == "purdue-fort-wayne":
            school_name = "ipfw"
        elif school_name == "omaha":
            school_name = "nebraska-omaha"
    except Exception as e:
        print(e)

    else:
        return school_name


def team_list(filepath):
    """
    Create dictionary of school names and formatted school names for mapping
    """
    team_names = pd.read_csv(filepath)
    school_list = team_names['school-format'].tolist()
    return school_list


def teams_dict(filepath, season):
    """
    Create dictionary of school names and formatted school names for mapping
    """
    filepath = filepath + str(season) + '.csv'
    team_names = pd.read_csv(filepath)
    team_names = team_names[['school', 'school-format']]
    team_dict = {}
    schools = team_names['school'].tolist()
    schools_format = team_names['school-format'].tolist()
    for school, schform in zip(schools, schools_format):
        team_dict[school] = schform
    return team_dict


def sos_dict_creator(filepath, season):
    """
    Create dictionary of school names and strength of schedule for mapping
    """
    filepath = filepath + str(season) + '.csv'
    team_sos = pd.read_csv(filepath)
    team_sos = team_sos[['school-format', 'sos']]
    sos_dict = {}
    schools = team_sos['school-format'].tolist()
    sos = team_sos['sos'].tolist()
    for school, sos in zip(schools, sos):
        sos_dict[school] = sos
    return sos_dict

def check_for_file(directory, filename):
    dir_files = os.listdir(directory)
    if filename in dir_files:
        print(f"{filename} already in {directory}")
        return True
    else:
        return False
