from flask import Flask, request, render_template, url_for
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn
import numpy as np
from scipy.stats import poisson,skellam
from flask import render_template

app = Flask(__name__)

epl_1617 = pd.read_csv("http://www.football-data.co.uk/mmz4281/1617/E0.csv")
epl_1617 = epl_1617[['HomeTeam','AwayTeam','FTHG','FTAG']]
epl_1617 = epl_1617.rename(columns={'FTHG': 'HomeGoals', 'FTAG': 'AwayGoals'})
epl_1617.head()

goal_model_data = pd.concat([epl_1617[['HomeTeam','AwayTeam','HomeGoals']].assign(home=1).rename(
            columns={'HomeTeam':'team', 'AwayTeam':'opponent','HomeGoals':'goals'}),
           epl_1617[['AwayTeam','HomeTeam','AwayGoals']].assign(home=0).rename(
            columns={'AwayTeam':'team', 'HomeTeam':'opponent','AwayGoals':'goals'})])

poisson_model = smf.glm(formula="goals ~ home + team + opponent", data=goal_model_data, 
                        family=sm.families.Poisson()).fit()


def simulate_match(foot_model, homeTeam, awayTeam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team': homeTeam, 
                                                            'opponent': awayTeam,'home':1},
                                                      index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team': awayTeam, 
                                                            'opponent': homeTeam,'home':0},
                                                      index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))




@app.route('/')

def hello(name=None):
    return render_template('index.html', name=name)

@app.route('/', methods=['POST','GET'])
def my_form_post():
      if request.method == "POST":
            sim = simulate_match(poisson_model, request.form['text'], request.form['text1'], max_goals=10)
            home_win = str(np.sum(np.tril(sim, -1)))
            draw_game = str(np.sum(np.diag(sim)))
            away_win = str(np.sum(np.triu(sim, 1)))
            print(home_win)
            return render_template('index.html', home_win = home_win, draw_game = draw_game, away_win = away_win)
      else:
            print("error")


if __name__ == '__main__':
      app.run(host='0.0.0.0')
