from flask import Flask, session, url_for, render_template, redirect
import joblib
from co2_form import Input_Form
import sys
import os
sys.path.append(os.getcwd() + "\model")
from model import modelo_final_concampos as mymodel

# creamos la app de Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

@app.route('/', methods=['GET','POST'])
def index():
    form = Input_Form()

    if form.validate_on_submit():
        session['Country'] = form.Country.data
        session['Year'] = form.Year.data
        session['GDP'] = form.GDP.data
        session['Population'] = form.Population.data
        session['Energy_production'] = form.Energy_production.data
        session['Energy_consumption'] = form.Energy_consumption.data
        session['CO2_emission'] = form.CO2_emission.data
        session['Energy_code'] = form.energy_type.data

        return redirect(url_for('prediction'))
    return render_template("home.html", form=form)

@app.route('/prediction', methods=['POST','GET'])
def prediction():

    results = mymodel.Final_Model.run_whole_model(
                            Country=session['Country'],
                            Year=session['Year'],
                            GDP=session['GDP'],
                            Population=session['Population'],
                            Energy_production=session['Energy_production'],
                            Energy_consumption=session['Energy_consumption'],
                            CO2_emission=session['CO2_emission'],
                            energy_type=session['Energy_code'])

    return render_template('prediction.html', results=results)

# Ejecutamos la aplicación app.run()
if __name__ == '__main__':
    # LOCAL
    # app.run(host='0.0.0.0', port=8080)

    # REMOTO
    app.run()
