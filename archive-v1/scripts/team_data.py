import pandas as pd

team18_url = 'https://www.sports-reference.com/cbb/seasons/2018-advanced-school-stats.html#adv_school_stats::none'
opp18_url = 'https://www.sports-reference.com/cbb/seasons/2018-advanced-opponent-stats.html#adv_opp_stats::none'
team17_url = 'https://www.sports-reference.com/cbb/seasons/2017-advanced-school-stats.html#adv_school_stats::none'
opp17_url = 'https://www.sports-reference.com/cbb/seasons/2017-advanced-opponent-stats.html#adv_opp_stats::none'
team16_url = 'https://www.sports-reference.com/cbb/seasons/2016-advanced-school-stats.html#adv_school_stats::none'
opp16_url = 'https://www.sports-reference.com/cbb/seasons/2016-advanced-opponent-stats.html#adv_opp_stats::none'
team15_url = 'https://www.sports-reference.com/cbb/seasons/2015-advanced-school-stats.html#adv_school_stats::none'
opp15_url = 'https://www.sports-reference.com/cbb/seasons/2015-advanced-opponent-stats.html#adv_opp_stats::none'
team14_url = 'https://www.sports-reference.com/cbb/seasons/2014-advanced-school-stats.html#adv_school_stats::none'
opp14_url = 'https://www.sports-reference.com/cbb/seasons/2014-advanced-opponent-stats.html#adv_opp_stats::none'
team13_url = 'https://www.sports-reference.com/cbb/seasons/2013-advanced-school-stats.html#adv_school_stats::none'
opp13_url = 'https://www.sports-reference.com/cbb/seasons/2013-advanced-opponent-stats.html#adv_opp_stats::none'

team18_df = pd.read_html(team18_url)[0]
opp18_df = pd.read_html(opp18_url)[0]
team17_df = pd.read_html(team17_url)[0]
opp17_df = pd.read_html(opp17_url)[0]
team16_df = pd.read_html(team16_url)[0]
opp16_df = pd.read_html(opp16_url)[0]
team15_df = pd.read_html(team15_url)[0]
opp15_df = pd.read_html(opp15_url)[0]
team14_df = pd.read_html(team14_url)[0]
opp14_df = pd.read_html(opp14_url)[0]
team13_df = pd.read_html(team13_url)[0]
opp13_df = pd.read_html(opp13_url)[0]

tmcols = ['Rk', 'School', 'G', 'W', 'L', 'W-L%', 'SRS', 'SOS', 'Conf-W',
'Conf-L', 'Home-W', 'Home-L', 'Away-W', 'Away-L', 'Points-for', 'Points-agnst',
'\xa0', 'Pace', 'Tm-ORtg', 'Tm-FTr', 'Tm-3PAr', 'Tm-TS%', 'Tm-TRB%', 'Tm-AST%',
'Tm-STL%', 'Tm-BLK%', 'Tm-eFG%', 'Tm-TOV%', 'Tm-ORB%', 'Tm-FT/FGA']

opcols = ['Rk', 'School', 'G', 'W', 'L', 'W-L%', 'SRS', 'SOS', 'Conf-W',
'Conf-L', 'Home-W', 'Home-L', 'Away-W', 'Away-L', 'Points-for', 'Points-agnst',
'\xa0', 'Pace', 'Op-ORtg', 'Op-FTr', 'Op-3PAr', 'Op-TS%', 'Op-TRB%', 'Op-AST%',
'Op-STL%', 'Op-BLK%', 'Op-eFG%', 'Op-TOV%', 'Op-ORB%', 'Op-FT/FGA']

def update_cols(tmdf, opdf, tmcols, opcols):
    tmdf.columns = tmcols
    opdf.columns = opcols
    tmdf = tmdf.drop(['\xa0'], axis=1)
    opdf = opdf.drop(['\xa0'], axis=1)
    tmdf = tmdf.drop(['Rk'], axis=1)
    opdf = opdf.drop(['Rk'], axis=1)
    opdf = opdf[['Op-ORtg', 'Op-FTr', 'Op-3PAr', 'Op-TS%', 'Op-TRB%', 'Op-AST%', 'Op-STL%', 'Op-BLK%', 'Op-eFG%', 'Op-TOV%', 'Op-ORB%', 'Op-FT/FGA']]
    tmdf = pd.concat([tmdf, opdf], axis=1)
    return tmdf

team18 = update_cols(team18_df, opp18_df, tmcols, opcols)
team17 = update_cols(team17_df, opp18_df, tmcols, opcols)
team16 = update_cols(team16_df, opp18_df, tmcols, opcols)
team15 = update_cols(team15_df, opp18_df, tmcols, opcols)
team14 = update_cols(team14_df, opp18_df, tmcols, opcols)
team13 = update_cols(team13_df, opp18_df, tmcols, opcols)

'''This works!  Now I need target data.  So I need to pull seed data and add seed & tourney columns.
In years before 2018 there is NCAA suffix on teams that made the tourney.

Goals:
Tourney picker,
seed picker, 
winner picker: ?
teams that pass first round, data: tourney rosters and matchups
'''


if __name__ == '__main__':
    print(team13.head())
