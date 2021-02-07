import pandas as pd
import os
from pathlib import Path

def read_clean_data():
    curr_filepath = os.path.split(__file__)[0]
    data_filepath = Path(curr_filepath).parents[0] / 'data/Heart_Disease_Mortality_Data_Among_US_Adults__35___by_State_Territory_and_County___2016-2018.csv'

    df = pd.read_csv(data_filepath)

    strats_df = df[(df.Stratification1!='Overall') | (df.Stratification2!='Overall')]

    counties = df[df.GeographicLevel == 'County']
    counties.shape

    t = counties.iloc[1,:]

    redux = counties[['LocationAbbr', 'LocationDesc',  
        'Data_Value', 'Data_Value_Footnote_Symbol', 'Data_Value_Footnote',
        'Stratification1', 'Stratification2', 'TopicID', 'LocationID', 'Y_lat',
        'X_lon']]

    redux.rename({'LocationAbbr':'state', 'LocationDesc':'county',  
        'Stratification1':'gender', 'Stratification2':'race'}, inplace=True)

    redux['y'] = redux.Data_Value / 100000

    return redux