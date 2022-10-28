# The_Bridge_CO2_ML_Project

## Goal:
A three phased machine learning project for predicting countries' co2 emissions and classifies them into beforehand made clusters

## Description:
From a dataset gathered from kaggle where we can find information about energy for the majority of the world countries we are trying to predict the co2 this countries are going to produce based on the specific characteristics of them, which have been clusterisized before the prediction, and after this prediction they are going to be classified into a category with the aim of developing a tool that enables us to simulate scenarios in base of the computed data of certain variables so the countries are going to be best informed about how to drive the future environmental policies or a base to assign public subsidies to the countries that behaves better, environmentally speaking.

### The Dataset:
In its origin, and after the EDA phase, the dataset is conformed of the following variables for the greatest energy producers:

- GDP: measured in ($ppp), base:2015.
- Population: measured in million people.
- Energy_production: measured in quad/btu.
- Energy_consumption: measured in quad/btu.
- co2_emission: Amount of co2 emitted, measured in co2 millions of tons.
- Per_capita_production: per capita production, measure in (quad/btu)/person.
- Energy_intensity_by_GDP: ineficiency of a country. Measured in (1000btu/(2015$ GDP PPP)).
- balance: difference between energy consumption and energy production.
- eficiency: co2 emitted by energy production, measured as c02 emission/energy production.
- energy_dependency: energy consumption by monetary unit, it tells us the energy dependency of a country, Measured as energy consumption/GDP.
- use_intensity_percapita: enrgy consumption per person in each country. Measured as energy consumption/population.
- co2_pc: it tells us how much co2 the country emits per person, it is measured by co2/population.
- Latitude: latitude
- Longitude: longitude
- Year: year when the data was taken
- Country: country name
- Energy_type: energy type
- Code_X: country's short code
- Continent: name of the continent the country belongs to
- Geometry: geometry of the country.

The data was obtained from Kaggle which got it from the US Energy administration. The data goes from the 1980 to the 2019

## Made Using:
- Python
- Visual Studio Code

## Libraries Used Mainly:
- Pandas
- Numpy
- Sklearn
- Statsmodels
- Plotly
- Matplotlib
- Seaborn

Author: Fernando Sánchez Olmo  
Upload date: 30-08-2022
