####
# race_imputer.py -- Imputes race from name and location based on predefined model and Census demographic data
####

import joblib
import pandas as pd
import numpy as np
import os

from yellowbrick.classifier import ClassificationReport, ROCAUC, PrecisionRecallCurve, ClassPredictionError

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, plot_roc_curve
from category_encoders import TargetEncoder

import matplotlib.pyplot as plt

DIR = os.path.dirname(__file__)

TEST_MODEL = False

model_string = ""
if TEST_MODEL: model_string = "2"



MODEL_COLS = ["first_name", "last_name", "middle_name", "white", "black", "indian", "asian", "other", "hispanic", "birth_year", "pctwhite_x", "pctblack_x", "pctapi_x", "pctaian_x", "pct2prace_x", "pcthispanic_x", "pctwhite_y", "pctblack_y", "pctapi_y", "pctaian_y", "pct2prace_y", "pcthispanic_y", "trigram_W", "trigram_B", "trigram_A", "trigram_I", "trigram_M", ]#"white_tx", "black_tx", "hispanic_tx", "asian_nyc", "black_nyc", "hispanic_nyc", "white_nyc"]


def train(MODEL="GNB"):

	# load voter data and merge with Census data
	df = pd.read_csv(DIR + "/data/nc_voter_geocoded_census_block_trigrams.csv")

	df = prep_data(df)

	tes = {}
	#tes = joblib.load(DIR + "/data/models/transformers_binary.joblib")

	models = {}


	# Loop through each race class, create model for each
	for race in  ["W", "B", "A", "I", "HL"]:

		X = df.copy()
		
		# If hispanic, use ethnic_code instead of race code
		if race == "HL":
			X["ethnic_code"] = np.where(X["ethnic_code"] == race, True, False)
			y = X["ethnic_code"]
			
		# other wise race code
		else:  
			X["race_code"] = np.where(X["race_code"] == race, True, False)
			y = X["race_code"]


		# target encode names, save target encoder
		for col in [ "first_name", "last_name", "middle_name"]:
			
			#te = tes[race][col]
			te = TargetEncoder()
			te.fit(X[col], y)
			
			X[col] = te.transform(X[col])

		# remove target variables and fill in any nas with 0
		#sample_weights = X["sample_weights"]
		#X = X.drop(["race_code", "ethnic_code", "zip", "sample_weights"], axis=1)
		X = X.fillna(0)

		sm = SMOTE(n_jobs=-1)
		X,y = sm.fit_resample(X, y)
		sample_weights = X["sample_weights"]
		X = X.drop(["zip", "sample_weights"], axis=1)

		# train model
		if MODEL == "LGBM":
			from lightgbm import LGBMClassifier
			model =  LGBMClassifier(n_jobs=-1)  
		elif MODEL == "GNB":
			from sklearn.naive_bayes import GaussianNB
			model = GaussianNB() 
		elif MODEL == "XGB":
			from xgboost import XGBClassifier
			model = XGBClassifier(n_jobs=-1)
		elif MODEL == "SGD":
			model = SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=True, l1_ratio=1.0, learning_rate="constant", loss="modified_huber", penalty="elasticnet", power_t=0.0)
		elif MODEL == "RF":
			from sklearn.ensemble import RandomForestClassifier
			model = RandomForestClassifier(n_jobs=-1, max_depth=10)

		model.fit(X[MODEL_COLS], y, sample_weight=sample_weights)

		# save model 
		models[race] = model

		# score model
		print(race, model.score(X[MODEL_COLS], y))

	# Save the models and encoders
	handle = MODEL.lower()

	#joblib.dump(tes, DIR + "/data/models/transformers_binary.joblib", compress=True)
	joblib.dump(models, DIR + "/data/models/models_binary_%s.joblib" % handle, compress=True)
	#joblib.dump(scalers, DIR + "/data/models/scalers_binary.joblib", compress=True)
	
	print("Trained model saved to ./data/models/")

def prep_census_location():
    """ Import and prep census data on race by census tract/block """

    # read in census data
    dfc = pd.read_csv(DIR + "/data/census_blocks/all.csv") 

    # defer to 0 for null
    dfc = dfc.fillna(0)

    # assert data type as int for location variables
    for col in ["census_tract", "block"]:
        dfc[col] = dfc[col].astype(int)

    # assert as float for all demographic percentage columns
    for col in list(dfc.columns):
        if col not in ["census_tract", "block"]:
            dfc[col] = dfc[col].astype(float)

    return dfc

def prep_census_location_zip():
    """ Import and prep census data on race by census tract/block """

    # read in census data
    dfc = pd.read_csv(DIR + "/data/census_zipcode.csv") 

    # defer to 0 for null
    dfc = dfc.fillna(0)

    # assert data type as int for location variables
    dfc["zip"] = dfc["zip"].astype(int)

    # assert as float for all demographic percentage columns
    for col in list(dfc.columns):
        if col != "zip":
            dfc[col] = dfc[col].astype(float)

    return dfc

def prep_census_name():
    """ import and prep Census data on race/ethnicity by last name """

    # read in census data
    dfc = pd.read_csv(DIR + "/data/census_names_merged.csv")

    # subset only the useful columns
    dfc = dfc[["name", "pctwhite", "pctblack", "pctapi", "pctaian", "pct2prace", "pcthispanic"]]

    # replace filler data with zero, set percetnages as floats
    for col in list(dfc.columns)[1:]:
        dfc[col] = dfc[col].replace("(S)", 0)
        dfc[col] = dfc[col].astype(float)

    return dfc
    

def append_trigrams(df):
    """ Merge trigram data to name data """

    # load trigram data
    trigrams = joblib.load(DIR + "/data/trigrams.joblib")

    # define apply function that gets all trigrams in a first name
    # and gets the score of each based on predefined score matrix
    # and sums them upp for an array of total scores for each race
    def get_trigram_score(x):
        scores = []
        for i in range(0, len(x.first_name) - 3):
            gram = x.first_name[i:i+3]
            if len(gram) > 2:
                if gram not in trigrams:
                    trigrams[gram] = [0]
                if len(scores) == 0:
                    scores = trigrams[gram]
                else:
                    scores = scores + trigrams[gram]
        return np.array(scores)/len(x.first_name)

    df['trigrams'] = df.apply(get_trigram_score, axis=1)

    # split the array of total scores into separate columns for each race class
    for i, race in enumerate(["W", "B", "A", "I", "M"]):
        df["trigram_" + race] = df.trigrams.apply(lambda x: x[i] if len(x) > i+1 else 0)

    return df

def prep_data(df):
    """
    Take in dataframe of names, locations, clean, format and merge with census data.
    """
    # Fill in blanks
    df["first_name"] = df.first_name.fillna("")
    df["last_name"] = df.last_name.fillna("")
    df["middle_name"] = df.middle_name.fillna("")

    # Convert all to uppercase
    df["middle_name"] = df.middle_name.apply(lambda x: x.upper())
    df["first_name"] = df.first_name.apply(lambda x: x.upper())
    df["last_name"] = df.last_name.apply(lambda x: x.upper())

    # eliminate suffixes from last names (jr, sr, III)
    def suff(x):
        tmp = x.last_name.split(" ")
        if tmp[-1] in ["JR", "I", "III", "II", "IV", "SR"]:
            return " ".join(tmp[0:-1])
        else:
            return x.last_name
    df["last_name"] = df.apply(suff, axis=1)

    # Get and merge in demographic data by census bloc/tract
    dfc = prep_census_location()
    df = df.merge(dfc, on=["census_tract", "block"], how="left")

    # get and merge demographic data by zip code when census tract-level data not available
    dfc = prep_census_location_zip()
    df.zip = df.zip.fillna("")
    df.zip = df.zip.apply(lambda x: str(x).split("-")[0])
    df.zip = df.zip.apply(lambda x: int(x) if x != "" else x)
    df = df.merge(dfc, on="zip", how="left")
    df["white"] = np.where(df.white_x.isna(), df.white_y, df.white_x)
    df["asian"] = np.where(df.asian_x.isna(), df.asian_y, df.asian_x)
    df["black"] = np.where(df.black_x.isna(), df.black_y, df.black_x)
    df["indian"] = np.where(df.indian_x.isna(), df.indian_y, df.indian_x)
    df["other"] = np.where(df.other_x.isna(), df.other_y, df.other_x)
    

    # get census name stats and merge on last and middle name
    dfc2 = prep_census_name()

    df = df.merge(dfc2, left_on="last_name", right_on="name", how="left")
    
    df = df.merge(dfc2, left_on="middle_name", right_on="name", how="left")
    
    # get and merge trigram data
    df = append_trigrams(df)

    return df

def census_predict(df, race):
    """ 
    use only census demographic probs to predict race 
    Average of probabilities for location, last name, middle name
    """
    if race == "A": 
        values = [df["asian"], df["pctapi_x"], df["pctapi_y"]]
    if race == "I": 
        values = [df["indian"], df["pctaian_x"], df["pctaian_y"]]
    if race == "HL": 
        values = [df["hispanic"], df["pcthispanic_x"], df["pcthispanic_y"]]
    if race == "W": 
        values = [df["white"], df["pctwhite_x"], df["pctwhite_y"]]
    if race == "B": 
        values = [df["black"], df["pctblack_x"], df["pctblack_y"]]

    # uses a simple average of proabilities to get overall probability  
    # potential for impovement here
    return sum(values)/len(values)

def impute(df, modelv="gnb", census=False):
    """ 
    Predict race/eth from a dataframe containing first_name, 
    last_name, middle_name, census_tract, block, birth_year 
    modelv - determines model version. gnb = Gaussian Naive Bayes, rf = Random Forest, xgb = XGBoost
    census - Toggles whether to use ML model or just census data to predict. ML models may not work well
    for minority groups with low population in NC dataset. False= only ML model. 
    True = ML model for W/B, census data otherwise
    """

    # load and merge all data
    df = prep_data(df)

    cols = ["first_name", "last_name", "middle_name", "census_tract", "block"]

    races = ["W", "B", "A", "I", "HL"]

    # load predefined model and transformers
    tes = joblib.load(DIR + "/data/models/transformers_binary.joblib")
    models = joblib.load(DIR + "/data/models/models_binary_%s%s.joblib" % (modelv, model_string))

    # loop thorugh each race/ethnicity class, transform and predict for each
    pred_cols = []
    for race in races:
        col2 = "race_pred_%s" % race

        # for certain race/ethnicities, NC data is unreliable. Defer to just census data
        # Fixed calculation for those probabilities
        if census and race in ["I", "A", "HL"]:
            df[col] = census_predict(df, race)

        else:
            # for model predicted columns, transform the data
            # using predefined transformers
            for col in [ "first_name", "last_name", "middle_name"]:
                te = tes[race][col]
                df[col] = te.transform(df[col])
                df[col] = df[col].fillna(0)

            # fill all nulls with zero
            df = df.fillna(0)
           
            # predict probabilities for each race
            preds = models[race].predict_proba(df[MODEL_COLS])

            # extract only the probabilities of true for that race/ethnicity as a new column
            df[col2] = [x[1] for x in preds]
            pred_cols.append(col2)

            df["race_pred2_%s" % race] = models[race].predict(df[MODEL_COLS])

    # use the highest probability as the prediction of race
    df["race_pred"] = df.apply(lambda x: races[list(x[pred_cols]).index(max(x[pred_cols]))], axis=1)
    
    return df

def impute_census(dfx):
    """
    Because of issues with the voter roll database, this alternate model will impute race 
    without that dataset based only on census statistics
    """
    dfx = prep_data(dfx)

    dfx = dfx.fillna(0)

    # get probabilities for eac race/ethnicity class

    dfx["pred_W"] = census_predict(dfx, "W")
    dfx["pred_A"] = census_predict(dfx, "A")
    dfx["pred_B"] = census_predict(dfx, "B")
    dfx["pred_I"] = census_predict(dfx, "I")
    dfx["pred_HL"] = census_predict(dfx, "HL")

    pred_cols = ["pred_W", "pred_B", "pred_A",  "pred_I", "pred_HL"]
    
    races = ["W", "B", "A", "I", "HL"]

    # Choose most likely race/ethnicity by max probability of individual class
    dfx["race_pred"] = dfx.apply(lambda x: races[list(x[pred_cols]).index(max(x[pred_cols]))], axis=1)

    return dfx


def evaluate(df, modelv="gnb", race="W", census=False, report=True, roc=True, pr=True):
    """ Run model evaluations for a specified model and race class """

    # get model
    models = joblib.load(DIR + "/data/models/models_binary_%s%s.joblib" % (modelv, model_string))
    model = models[race]

    # get data
    df = prep_data(df)
    tes = joblib.load(DIR + "/data/models/transformers_binary.joblib")

    # transform data
    for col in [ "first_name", "last_name", "middle_name"]:
        te = tes[race][col]
        df[col] = te.transform(df[col])
        df[col] = df[col].fillna(0)

    tmpa = np.where(df.race_code == race, True, False)
    df = df.fillna(0)

    # run specified evaluation visualizer
    if report:
        visualizer = ClassificationReport(model, classes=model.classes_, support=True)
        visualizer.score(df[MODEL_COLS], tmpa)
        visualizer.show() 

    if roc:
        visualizer = ROCAUC(model, classes=["W", "not-W"])
        visualizer.score(df[MODEL_COLS], tmpa)
        visualizer.show()
    if pr:
        viz = PrecisionRecallCurve(model, is_fitted=True, classes=["W", "not-W"])
        viz.score(df[MODEL_COLS], tmpa)
        viz.show()



def eval_models(df, race="W", models=["gnb", "rf", "xgb"], census=False, report=False, roc=False, pr=False, cpe=False):

    """ Run evaluation on a set of models and a single race class """

    df = prep_data(df)
    tes = joblib.load(DIR + "/data/models/transformers_binary.joblib")

    for col in [ "first_name", "last_name", "middle_name"]:
        te = tes[race][col]
        df[col] = te.transform(df[col])
        df[col] = df[col].fillna(0)

    tmpa = np.where(df.race_code == race, True, False)
    df = df.fillna(0)

    for modelv in models:

        models = joblib.load(DIR + "/data/models/models_binary_%s%s.joblib" % (modelv, model_string))
        model = models[race]
        
        model.target_type_ = "binary"

        if report:
            visualizer = ClassificationReport(model, classes=model.classes_, support=True)
            visualizer.score(df[MODEL_COLS], tmpa)
            visualizer.show() 

        if roc:
            visualizer = ROCAUC(model, classes=["W", "not-W"])
            visualizer.score(df[MODEL_COLS], tmpa)
            visualizer.show()
        if pr:
            viz = PrecisionRecallCurve(model, is_fitted=True, classes=["W", "not-W"])
            viz.score(df[MODEL_COLS], tmpa)
            viz.show()

        if cpe:
            viz = ClassPredictionError(model)
            viz.score(df[MODEL_COLS], tmpa)
            viz.show()



def roc_curves(df, modelv="gnb",  census=False):
    """ Create a single graphic of all ROC Curves with 
    every race/ethnicty class for a model """

    # get model
    models = joblib.load(DIR + "/data/models/models_binary_%s%s.joblib" % (modelv, model_string))

    # get all data
    df = prep_data(df)
    tes = joblib.load(DIR + "/data/models/transformers_binary.joblib") 
    ax = plt.gca()

    # loop through each race/eth and plot roc
    for race in ["W", "B", "A", "I", "HL"]:
        # load model
        model = models[race]
        X = df.copy()

        # transform data
        for col in [ "first_name", "last_name", "middle_name"]:
            te = tes[race][col]
            X[col] = te.transform(df[col])
            X[col] = X[col].fillna(0)

        tmpa = np.where(X.race_code == race, True, False)
        X = X.fillna(0)

        # plot roc
        plot_roc_curve(model, X[MODEL_COLS], tmpa, ax=ax, alpha=.8)




