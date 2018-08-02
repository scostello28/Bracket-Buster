import pandas as pd

team_names = pd.read_csv('team_list/sos_team_list_2018_final.csv')

team_names = team_names[['School', 'School_format']]

team_dict = {}

schools = team_names['School'].tolist()
schools_format = team_names['School_format'].tolist()

for school, schform in zip(schools, schools_format):
    team_dict[school] = schform
