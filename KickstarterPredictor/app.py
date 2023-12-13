import pandas as pd
from flask import Flask, request, render_template
import pickle
import xgboost as xgb

# initialise the flask
app = Flask(__name__)
model = pickle.load(open('finalized_model.sav', 'rb'))


# http://127.0.0.1:5000/

# define html file to get user input
@app.route('/')
def home():
    return render_template("predict.html")


# prediction function
@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML
    """

    goal_usd = request.form['goal_usd']
    country_US = request.form['country_US']
    cam_duration = request.form['cam_duration']
    prep_duration = request.form['prep_duration']
    description_length = request.form['description_length']
    staff_pick_True = request.form['staff_pick_True']
    Comics, Crafts, Dance, Design, Fashion, Film_Video, Food = 0, 0, 0, 0, 0, 0, 0
    Games, Journalism, Music, Photography, Publishing, Technology, Theater = 0, 0, 0, 0, 0, 0, 0
    # category
    category_list = ['Comics', 'Crafts', 'Dance', 'Design', 'Fashion', 'Film_Video', 'Food',
                     'Games', 'Journalism', 'Music', 'Photography', 'Publishing', 'Technology', 'Theater']
    input_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    category_input = request.form["category_input"]
    if category_input in category_list:
        input_list[category_list.index(category_input)] = 1

    feature_dictionary = {
        Comics: 'Comics', Crafts: 'Crafts',
        Dance: 'Dance', Design: 'Design',
        Fashion: 'Fashion', Film_Video: 'Film_Video',
        Food: 'Food', Games: 'Games', Journalism: 'Journalism',
        Music: 'Music', Photography: 'Photography',
        Publishing: 'Publishing', Technology: 'Technology', Theater: 'Theater'
    }

    i = 0
    for item in category_list:
        feature_dictionary[item] = input_list[i]
        i = i + 1

    input_variables = pd.DataFrame([[goal_usd, country_US, cam_duration,
                                     prep_duration, description_length, staff_pick_True, Comics, Crafts, Dance, Design,
                                     Fashion, Film_Video, Food, Games, Journalism, Music, Photography, Publishing,
                                     Technology, Theater]], columns=['goal_usd', 'country_US', 'cam_duration',
                                                                     'prep_duration', 'description_length', 'staff_pick_True',
                                                                     'Comics', 'Crafts',
                                                                     'Dance', 'Design', 'Fashion', 'Film_Video', 'Food',
                                                                     'Games', 'Journalism',
                                                                     'Music', 'Photography', 'Publishing', 'Technology',
                                                                     'Theater'], dtype=float)

    result = model.predict(input_variables)

    if int(result) == 1:
        prediction = 'succeed!'
        suggestion = 'Great! Congratulations!'
    else:
        prediction = 'not succeed :( '
        suggestion = 'Please contact us to see how you can increase your chances!'
    return render_template("result.html", prediction=prediction, suggestion=suggestion)


# main function
if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
