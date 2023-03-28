import os
import pandas as pd 
import numpy as np
import env
from sklearn.model_selection import train_test_split
import sklearn


protocol = ' '
db = ' '
user =''
password = ' '
host = ' '
mysqlcon = f"{protocol}://{user}:{password}@{host}/{db}"

######################## Zillow Data ###########
def get_connection(db, user, host, password, protocol):
    return f'{protocol}://{user}:{password}@{host}/{db}'
   

def get_zillow_data():
    filename = "zillow_exc.csv"
    mysqlcon=f"{protocol}://{user}:{password}@{host}/zillow"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql_query('''select bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
                                taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid,
                                propertylandusedesc
                                from properties_2017
                                left join propertylandusetype
                                using (propertylandusetypeid)''', mysqlcon)

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df


def wrangle_zillow(df):
    '''Wrangle zillow will clean the data and update missing values 
    for the df'''
   #Replace a whitespace sequence or empty with a NaN value and reassign this manipulation to df.
    df = df.replace(r'^\s*$', np.nan, regex=True)    
    #drop any duplicates in the df
    df = df.drop_duplicates()
    #filter column for single family homes
    df = df[df['propertylandusetypeid']==261.0]
    #drop redundant column
    df = df.drop(columns = 'propertylandusedesc')
    #filled nan values with 0 
    df = df.fillna(0)
    return df

###### Zillow Prepare #######
####### Split data #############
def train_test_split(df):
    x_train_and_validate, x_test = train_test_split(df, random_state=123)
    x_train, x_validate = train_test_split(x_train_and_validate)
    return x_train, x_validate, x_test, x_train_and_validate

################# Data Scalers ##################
def minmax_scaler():
    ######## Min Max Scaler (range calculations)
    scaler = sklearn.preprocessing.MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)
    ### Apply to train, validate, and test
    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_validate_scaled, x_test_scaled

def standard_scaler():
    scaler = sklearn.preprocessing.StandardScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)
    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return x_train_scaled, x_validate_scaled, x_test_scaled

def robust_scaler():
    scaler = sklearn.preprocessing.RobustScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    plt.figure(figsize=(13, 6))
    plt.subplot(121)
    plt.hist(x_train, bins=25, ec='black')
    plt.title('Original')
    plt.subplot(122)
    plt.hist(x_train_scaled, bins=25, ec='black')
    plt.title('Scaled')
    plt.show()
    return x_train_scaled, x_validate_scaled, x_test_scaled


#############Student Data

def get_student_data():
    filename = "student_grades.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('SELECT * FROM student_grades', get_connection('school_sample'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def wrangle_grades():
    '''
    Read student_grades into a pandas DataFrame from mySQL,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''

    # Acquire data

    grades = get_student_data()

    # Replace white space values with NaN values.
    grades = grades.replace(r'^\s*$', np.nan, regex=True)

    # Drop all rows with NaN values.
    df = grades.dropna()

    # Convert all columns to int64 data types.
    df = df.astype('int')

    return df
