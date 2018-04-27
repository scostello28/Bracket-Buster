import pandas as pd

sos = pd.read_csv('sos_team_list_2018.csv')

def format_schools(row):
    row['School'] = row['School'].rstrip("-")
    row['School_format'] = row['School'].lower()
    row['School_format'] = row['School_format'].replace("&",'')
    row['School_format'] = row['School_format'].replace(" ", "-")
    row['School_format'] = row['School_format'].replace("--", "-")
    row['School_format'] = row['School_format'].replace("'",'')
    row['School_format'] = row['School_format'].replace(".",'')
    row['School_format'] = row['School_format'].replace("(",'')
    row['School_format'] = row['School_format'].replace(")",'')
    row['School_format'] = row['School_format'].rstrip("-")
    return row

sos = sos.apply(format_schools, axis=1)

sos.to_csv('sos_team_list_2018_updated.csv')
