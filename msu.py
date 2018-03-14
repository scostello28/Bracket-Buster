import pandas as pd
import numpy as np

# teamsurl = 'https://www.sports-reference.com/cbb/schools/michigan-state/2018-gamelogs.html#sgl-basic::none'

# teams = ['north-florida', 'michigan-state', 'duke', 'stony-brook', 'depaul',
#          'connecticut', 'north-carolina', 'notre-dame']
# teams = ['michigan', 'wisconsin']

# teams = ['abilene-christian', 'air-force', 'akron', 'alabama-am', 'alabama', 'alabama-state', 'alabama-birmingham', 'albany-ny', 'alcorn-state', 'american']

# teams = ['appalachian-state', 'arizona-state', 'arizona', 'arkansas', 'arkansas-state']
# teams = ['arkansas-little-rock', 'arkansas-pine-bluff', 'army', 'auburn']
# teams = ['austin-peay', 'ball-state', 'baylor', 'belmont', 'bethune-cookman', 'binghamton', 'boise-state', 'boston-college', 'boston-university', 'bowling-green-state', 'bradley', 'brigham-young', 'brown', 'bryant', 'bucknell', 'buffalo', 'butler', 'cal-poly', 'cal-state-bakersfield']
# teams = ['cal-state-fullerton', 'cal-state-northridge', 'campbell', 'canisius', 'central-arkansas', 'central-connecticut-state', 'central-florida', 'central-michigan', 'charleston-southern', 'charlotte', 'chattanooga', 'chicago-state', 'cincinnati', 'citadel', 'clemson']
# teams = ['cleveland-state', 'coastal-carolina', 'colgate', 'college-of-charleston', 'colorado', 'colorado-state', 'columbia', 'connecticut', 'coppin-state', 'cornell', 'creighton', 'dartmouth', 'davidson', 'dayton', 'delaware', 'delaware-state', 'denver', 'depaul', 'detroit-mercy']
# teams = ['drake', 'drexel', 'duke', 'duquesne', 'east-carolina', 'east-tennessee-state', 'eastern-illinois', 'eastern-kentucky', 'eastern-michigan', 'eastern-washington', 'elon', 'evansville', 'fairfield', 'fairleigh-dickinson']
# teams = ['florida-am', 'florida-atlantic', 'florida', 'florida-gulf-coast', 'florida-international', 'florida-state']
# teams = ['fordham', 'fresno-state', 'furman', 'gardner-webb', 'george-mason', 'george-washington', 'georgetown', 'georgia']
# teams = ['georgia-southern', 'georgia-state', 'georgia-tech', 'gonzaga', 'grambling', 'grand-canyon', 'green-bay', 'hampton', 'hartford', 'harvard', 'hawaii', 'high-point', 'hofstra', 'holy-cross', 'houston-baptist', 'houston', 'howard', 'idaho-state', 'idaho']
# teams = ['illinois', 'illinois-state', 'illinois-chicago', 'incarnate-word', 'indiana', 'indiana-state', 'iona', 'iowa', 'iowa-state', 'ipfw', 'iupui', 'jackson-state', 'jacksonville', 'jacksonville-state', 'james-madison', 'kansas', 'kansas-state', 'kennesaw-state', 'kent-state']
# teams = ['kentucky', 'la-salle', 'lafayette', 'lamar', 'lehigh', 'liberty', 'lipscomb', 'long-beach-state']
# teams = ['long-island-university']
# teams =  ['longwood']
# teams = ['louisiana-lafayette']
# teams = ['louisiana-state']
# teams = ['louisiana-tech', 'louisiana-monroe', 'louisville']
# teams = ['loyola-il', 'loyola-md', 'loyola-marymount', 'maine', 'manhattan', 'marist', 'marquette', 'marshall', 'maryland', 'maryland-baltimore-county', 'maryland-eastern-shore', 'massachusetts', 'massachusetts-lowell']
# teams = ['mcneese-state', 'memphis', 'mercer', 'miami-fl', 'miami-oh', 'michigan-state', 'michigan', 'middle-tennessee', 'milwaukee', 'minnesota', 'mississippi', 'mississippi-state', 'mississippi-valley-state', 'missouri-state', 'missouri']
# teams = ['missouri-kansas-city', 'monmouth', 'montana', 'montana-state', 'morehead-state', 'morgan-state', 'mount-st-marys', 'murray-state', 'navy', 'nebraska', 'nebraska-omaha', 'nevada', 'nevada-las-vegas', 'new-hampshire', 'new-mexico', 'new-mexico-state', 'new-orleans', 'niagara', 'nicholls-state']
# teams = ['njit', 'norfolk-state', 'north-carolina-at', 'north-carolina-central', 'north-carolina-state', 'north-carolina', 'north-carolina-asheville', 'north-carolina-greensboro', 'north-carolina-wilmington', 'north-dakota-state', 'north-dakota', 'north-florida', 'north-texas', 'northeastern']
# teams = ['northern-arizona', 'northern-colorado', 'northern-illinois', 'northern-iowa', 'northern-kentucky', 'northwestern-state', 'northwestern', 'notre-dame', 'oakland', 'ohio', 'ohio-state', 'oklahoma', 'oklahoma-state', 'old-dominion', 'oral-roberts', 'oregon', 'oregon-state', 'pacific', 'penn-state']
# teams = ['pennsylvania', 'pepperdine', 'pittsburgh', 'portland', 'portland-state', 'prairie-view', 'presbyterian', 'princeton', 'providence', 'purdue', 'quinnipiac', 'radford', 'rhode-island', 'rice', 'richmond', 'rider']
# teams = ['robert-morris', 'rutgers', 'sacramento-state', 'sacred-heart', 'saint-francis-pa', 'saint-josephs', 'saint-louis', 'saint-marys-ca', 'saint-peters', 'sam-houston-state', 'samford', 'san-diego-state']
# teams = ['san-diego', 'san-francisco', 'san-jose-state', 'santa-clara', 'savannah-state', 'seattle', 'seton-hall', 'siena', 'south-alabama', 'south-carolina', 'south-carolina-state', 'south-carolina-upstate', 'south-dakota', 'south-dakota-state', 'south-florida']
# teams = ['southeast-missouri-state', 'southeastern-louisiana', 'southern-california', 'southern-illinois', 'southern-illinois-edwardsville', 'southern', 'southern-methodist', 'southern-mississippi', 'southern-utah']
# teams = ['st-bonaventure', 'st-francis-ny', 'st-johns-ny', 'stanford', 'stephen-f-austin', 'stetson', 'stony-brook', 'syracuse', 'temple', 'tennessee-state', 'tennessee-tech', 'tennessee', 'tennessee-martin']
# teams = ['texas-am', 'texas-am-corpus-christi', 'texas-christian', 'texas']
# teams = ['texas-southern', 'texas-state', 'texas-tech', 'texas-arlington']
# teams = ['texas-el-paso']
# teams = ['texas-pan-american']
# teams = ['texas-san-antonio', 'toledo']
# teams = ['towson', 'troy', 'tulane', 'tulsa', 'california-davis']
# teams = ['california-irvine', 'california-riverside', 'california-santa-barbara', 'ucla', 'california', 'utah-state', 'utah', 'utah-valley', 'valparaiso']
# teams = ['vanderbilt', 'vermont', 'villanova', 'virginia', 'virginia-commonwealth', 'virginia-military-institute', 'virginia-tech', 'wagner', 'wake-forest', 'washington', 'washington-state', 'weber-state', 'west-virginia']
# teams = ['western-carolina', 'western-illinois', 'western-kentucky', 'western-michigan', 'wichita-state', 'william-mary', 'winthrop', 'wisconsin', 'wofford', 'wright-state', 'wyoming', 'xavier', 'yale', 'youngstown-state']



team_names_sos_filepath = 'team_list/sos_team_list_2018_final.csv'

def team_list(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    school_list = team_names['School_format'].tolist()
    return school_list

def teams_dict(filepath):
    '''
    Create dictionary of school names and formatted school names for mapping
    '''
    team_names = pd.read_csv(filepath)
    team_names = team_names[['School', 'School_format']]
    team_dict = {}
    schools = team_names['School'].tolist()
    schools_format = team_names['School_format'].tolist()
    for school, schform in zip(schools, schools_format):
        team_dict[school] = schform
    return team_dict

def sos_dict(filepath):
    '''
    Create dictionary of school names and strengthof schedule for mapping
    '''
    team_sos = pd.read_csv(filepath)
    team_sos = team_sos[['School_format', 'SOS']]
    sos_dict = {}
    schools = team_sos['School_format'].tolist()
    sos = team_sos['SOS'].tolist()
    for school, sos in zip(schools, sos):
        sos_dict[school] = sos
    return sos_dict


def df_creator(teams, season):
    '''
    INPUTs:
        teams = list of teams (formatted as in url)
        season = season year

    OUTPUT: DataFrame of all games
    '''
    games_df = pd.DataFrame()

    for team in teams:

        url = 'https://www.sports-reference.com/cbb/schools/{}/{}-gamelogs.html#sgl-basic::none'.format(team, season)

        '''Read team gamelog'''
        df = pd.read_html(url)[0]

        '''remove oppenent columns'''
        df = df.iloc[:, 0:23]

        '''Remove divider rows'''
        df = df.drop(df.index[[20,21]])

        '''Remove Double Row headers'''
        dubcols = df.columns.tolist()
        cols = [col[1] for col in dubcols]
        df.columns = cols

        '''Rename Columns'''
        newcols = ['G', 'Date', 'Blank', 'Opp', 'W', 'Pts', 'PtsA', 'FG', 'FGA',
                   'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                   'AST', 'STL', 'BLK', 'TO', 'PF']
        df.columns = newcols

        '''reformat Opp column strings'''
        # df['Opp'] = df['Opp'].str.lower()
        # df['Opp'] = df['Opp'].str.replace("'",'')
        # df['Opp'] = df['Opp'].str.replace(".",'')
        # df['Opp'] = df['Opp'].str.replace("(",'')
        # df['Opp'] = df['Opp'].str.replace(")",'')
        # df['Opp'] = df['Opp'].str.replace(" ",'-')
        df['Opp'] = df['Opp'].map(teams_dict(team_names_sos_filepath))

        '''Only take the first charcter in W field then map to 0's and 1's'''
        df['W'] = df['W'].astype(str).str[0]
        df['W'] = df['W'].map({'W': 1, 'L': 0})

        '''Create win precentage and rolling average Features'''
        df['Ws'] = df['W'].cumsum(axis=0)
        df['Wp'] = df['Ws'].astype(int) / df['G'].astype(int)
        df['ppg'] = df['Pts'].rolling(window=5,center=False).mean()
        df['pApg'] = df['PtsA'].rolling(window=5,center=False).mean()
        df['FGp'] = df['FG%'].rolling(window=5,center=False).mean()
        df['3Pp'] = df['3P%'].rolling(window=5,center=False).mean()
        df['FTp'] = df['FT%'].rolling(window=5,center=False).mean()
        df['ORBpg'] = df['ORB'].rolling(window=5,center=False).mean()
        df['RBpg'] = df['RB'].rolling(window=5,center=False).mean()
        df['ASTpg'] = df['AST'].rolling(window=5,center=False).mean()
        df['STLpg'] = df['STL'].rolling(window=5,center=False).mean()
        df['BLKpg'] = df['BLK'].rolling(window=5,center=False).mean()
        df['TOpg'] = df['TO'].rolling(window=5,center=False).mean()
        df['PFpg'] = df['PF'].rolling(window=5,center=False).mean()

        '''Remove columns after rolling ave calcs'''
        df = df.drop(['G', 'Blank', 'Pts', 'PtsA', 'FG', 'FGA', 'FG%',
                      '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'RB',
                      'AST', 'STL', 'BLK', 'TO', 'PF'], axis=1)

        '''Drop NaN rows before rolling averages can be calc'd'''
        df = df.dropna()

        '''Add Team Column'''
        df['Tm'] = team

        '''Add SOS columns'''
        df['sos'] = df['Tm'].map(sos_dict(team_names_sos_filepath))

        '''Add df to games_df'''
        games_df = games_df.append(df, ignore_index=True)

    return games_df

def get_unique_id(row):
    '''
    Create matchup and ID rows
    '''
    row['matchup'] = ",".join(sorted([row['Tm'], row['Opp']]))
    row['ID'] = '{},{}'.format(row['matchup'], row['Date'])
    return row

def everybody_merge(df):
    '''
    INPUT: DataFrame
    OUTPUT: DataFrame with matching IDs merged to same row
    '''

    '''Add cumulative conditional count column'''
    df['count'] = df.groupby('ID').cumcount() + 1

    '''Create separate dataframes for 1st and 2nd instances of games'''
    df1 = df[df['count'] == 1]
    df2 = df[df['count'] == 2]

    '''Select needed columns from 2nd instance DataFrame and
    rename te prepare for pending left merge'''
    df2_stats = df2.iloc[:, 5:19]
    df2_id = df2['ID']
    g2cols = df2_stats.columns.tolist()
    OPcols = ['OP{}'.format(col) for col in g2cols]
    df2_stats.columns = OPcols
    df2 = pd.concat([df2_stats, df2_id], axis=1)

    '''Merge games instance DataFrames'''
    df = pd.merge(df1, df2, how='left', on='ID')

    '''Drop redundant Opp column and any games where there is no data
    for oppenent'''
    df = df.drop(['Opp'], axis=1)
    df = df.dropna()

    return df




if __name__ == '__main__':
    # print(df_creator(teams, 2018))
    filepath = 'team_list/sos_team_list_2018_final.csv'
    teams = team_list(filepath)
    games = df_creator(teams, 2018)
    games = games.apply(get_unique_id, axis=1)
    games = everybody_merge(games)
