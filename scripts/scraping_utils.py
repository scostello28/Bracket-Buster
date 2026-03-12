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

        school_name_map = {
            "siu-edwardsville": "southern-illinois-edwardsville",
            "vmi": "virginia-military-institute",
            "uc-davis": "california-davis",
            "uc-irvine": "california-irvine",
            "uc-riverside": "california-riverside",
            "uc-santa-barbara": "california-santa-barbara",
            "uc-san-diego": "california-san-diego",
            "university-of-california": "california",
            "cal-state-long-beach": "long-beach-state",
            "louisiana": "louisiana-lafayette",
            "texas-rio-grande-valley": "texas-pan-american",
            "byu": "brigham-young",
            "etsu": "east-tennessee-state",
            "liu": "long-island-university",
            "lsu": "louisiana-state",
            "nc-state": "north-carolina-state",
            "ole-miss": "mississippi",
            "pitt": "pittsburgh",
            "penn": "pennsylvania",
            "saint-marys": "saint-marys-ca",
            "smu": "southern-methodist",
            "tcu": "texas-christian",
            "umbc": "maryland-baltimore-county",
            "umass": "massachusetts",
            "umass-lowell": "massachusetts-lowell",
            "unc": "north-carolina", 
            "unc-asheville": "north-carolina-asheville",
            "unc-greensboro": "north-carolina-greensboro",
            "unc-wilmington": "north-carolina-wilmington",
            "ucf": "central-florida",
            "uab": "alabama-birmingham",
            "the-citadel": "citadel",
            "ucsb": "california-santa-barbara",
            "uconn": "connecticut",
            "umkc": "missouri-kansas-city",
            "unlv": "nevada-las-vegas",
            "utep": "texas-el-paso",
            "utsa": "texas-san-antonio",
            "ut-arlington": "texas-arlington",
            "usc": "southern-california",
            "vcu": "virginia-commonwealth",
            "uic": "illinois-chicago",
            "little-rock": "arkansas-little-rock",
            "purdue-fort-wayne": "ipfw",
            "omaha": "nebraska-omaha",
            "bowling-green": "bowling-green-state",
            "east-texas-am": "texas-am-commerce",
            "fdu": "fairleigh-dickinson",
            "houston-christian": "houston-baptist",
            "iu-indy": "iupui",
            "kansas-city": "missouri-kansas-city",
            "sam-houston": "sam-houston-state",
            "st-thomas": "st-thomas-mn",
            "utah-tech": "dixie-state"
        }

        if school_name in school_name_map.keys():
            school_name = school_name_map[school_name]

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
