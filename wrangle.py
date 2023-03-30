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

###### Zillow Prepare ######################################
####### Split data #########################################
def train_test_split(df, random_stat=123):
    x_train_and_validate, x_test = train_test_split(df, random_state=123)
    x_train, x_validate = train_test_split(x_train_and_validate)
    return x_train, x_validate, x_test, x_train_and_validate

################# Data Scalers ###########################################
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


#############Student Data#####################################

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

'''Wrangles data from Zillow Database'''

##################################################Wrangle.py###################################################

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import user, password, host

#**************************************************Acquire*******************************************************

def acquire_zillow():
    ''' Acquire data from Zillow using env imports and rename columns'''
    
    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"
    
    query = """
            
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017

    LEFT JOIN propertylandusetype USING(propertylandusetypeid)

    WHERE propertylandusedesc IN ("Single Family Residential",                       
                                  "Inferred Single Family Residential")"""

    # get dataframe of data
    df = pd.read_sql(query, url)
    
    
    # renaming column names to one's I like better
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'area',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'year_built',})
    return df

#**************************************************Remove Outliers*******************************************************

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

#**************************************************Distributions*******************************************************

def get_hist(df):
    ''' Gets histographs of acquired continuous variables'''
    
    plt.figure(figsize=(16, 3))

    # List of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display histogram for column.
        df[col].hist(bins=5)

        # Hide gridlines.
        plt.grid(False)

        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)

        plt.tight_layout()

    plt.show()
        
        
def get_box(df):
    ''' Gets boxplots of acquired continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):

        # i starts at 0, but plot should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[[col]])

        # Hide gridlines.
        plt.grid(False)

        # sets proper spacing between plots
        plt.tight_layout()

    plt.show()
        
#**************************************************Prepare*******************************************************

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount'])
    
    # get distributions of numeric data
    get_hist(df)
    get_box(df)
    
    # converting column datatypes
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using mode
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test    


#**************************************************Wrangle*******************************************************


def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test