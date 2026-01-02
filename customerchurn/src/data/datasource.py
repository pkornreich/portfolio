# BS"D

import logging

import pandas as pd
from pandas import DataFrame #, Series
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from constants import Constants
from util.logging import Logger
from models.modelobjects import ModelData
from data.datamanager import DataManager

class DataService:

    def __init__(self, file_name: str = None):
        self.is_experiment = False
        if file_name is None or len(file_name) == 0:
            self.df = DataService.read_dataset()
            self.is_experiment = True
        else:
            self.df = DataService.read_input(file_name)
        self.cleansed_df = None
        self.data_description = 'Base Data'
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'numAdminTickets', 'numTechTickets']

    
    @staticmethod
    def read_dataset() -> DataFrame:
        # Determine if running from root (e.g. Pytest/PROD?) or running from src
        # Will affect data file location.
        full_path = Constants.DATA_PATH + Constants.DATA_FILE_NAME
        if DataManager.is_running_from_src():
            full_path = Constants.PATH_TO_ROOT + full_path
        df: DataFrame = pd.read_excel(full_path)
        return df
    
    @staticmethod
    def read_input(file_name: str = None) -> DataFrame:
        return pd.read_json(file_name)
    
    @staticmethod
    def read_default_values() -> DataFrame:
        default_values_file = Constants.LOG_PATH + Constants.DEFAULT_VALUES_FILE_NAME
        if DataManager.is_running_from_src():
            default_values_file = Constants.PATH_TO_ROOT + default_values_file
        return pd.read_json(default_values_file)
    
    def save_default_values(self):
        default_values: map = {}
        column_names = list(self.df.columns)
        column_names.remove('customerID')
        column_names.remove('Churn')
        for name in column_names:
            if name in self.numeric_columns:
                default_values[name] = self.df[name].mean()
            else:
                default_values[name] = self.df[name].mode()[0]
        # ensure int values are int
        default_values["numAdminTickets"] = round(default_values["numAdminTickets"])
        default_values["numTechTickets"] = round(default_values["numTechTickets"])
        default_values["tenure"] = round(default_values["tenure"])
        default_values_file = Constants.LOG_PATH + Constants.DEFAULT_VALUES_FILE_NAME
        if DataManager.is_running_from_src():
            default_values_file = Constants.PATH_TO_ROOT + default_values_file
        df: DataFrame = DataFrame([default_values])
        df.to_json(default_values_file, orient='records', indent=4, index=False)
        
    def convert_binary_values(self):
        # Gender
        self.df['gender'] = self.df['gender'].replace({'Female': 1, 'Male': 0})
        # Simple Yes/No columns
        simple_binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
        for column in simple_binary_columns:
            self.df[column] = self.df[column].replace({'Yes': 1, 'No': 0})
        # Could be prediction, in which case, Churn may be missing - so only update when experimenting
        if self.is_experiment:
            self.df['Churn'] = self.df['Churn'].replace({'Yes': 1, 'No': 0})

        # Phone Service
        self.df['MultipleLines'] = self.df['MultipleLines'].replace({'Yes': 1, 'No': 0, 'No phone service': 0})
        # Internet Service
        internet_binary_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                   'TechSupport', 'StreamingTV', 'StreamingMovies']
        for column in internet_binary_columns:
            self.df[column] = self.df[column].replace({'Yes': 1, 'No': 0, 'No internet service': 0})


    # def scale_columns(self):
    #     # Scale Appropriate Columns
    #     scaler = self.get_scaler()
    #     self.cleansed_df['tenure'] = scaler.fit_transform(self.cleansed_df[['tenure']])
    #     self.cleansed_df['MonthlyCharges'] = scaler.fit_transform(self.cleansed_df[['MonthlyCharges']])
    #     self.cleansed_df['TotalCharges'] = scaler.fit_transform(self.cleansed_df[['TotalCharges']])
    #     self.cleansed_df['numAdminTickets'] = scaler.fit_transform(self.cleansed_df[['numAdminTickets']])
    #     self.cleansed_df['numTechTickets'] = scaler.fit_transform(self.cleansed_df[['numTechTickets']])


    def cleanse_data(self):
        # Change appropriate string columns to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')

        if self.is_experiment:
            # Handle Missing Values
            null_count = self.df['TotalCharges'].isna().sum()
            non_null_count = self.df['TotalCharges'].count()
            message = f" Percentage of Null Total Charges Values: {round(null_count*100/(non_null_count + null_count), 2)}%"
            # Shows non-null count is < 1%
            logger = Logger.get_logger()
            logger.log(logging.INFO, message)
            # So, can drop them
            self.df.dropna(subset=['TotalCharges'], inplace=True)
        
            # Now, that any problematic rows have been dropped, can get default values
            # Also need to do this before converting to other values.
            self.save_default_values()

        # Binary Value Columns
        self.convert_binary_values()
        # Convert naturally ordered String columns to numeric
        self.df['Contract'] = self.df['Contract'].replace({'Month-to-month': 0,  'One year': 1, 'Two year': 2})

        # Multi-value string columns that don't have a natural order - One-Hot Encode
        df_encoded = None
        # For experiment, can do automaticallly
        if self.is_experiment:
            df_encoded = pd.get_dummies(self.df, columns=['InternetService', 'PaymentMethod'])
        # For prediction, have to know something about possible values. . .
        else:
            is_dummy_columns = ['InternetService_DSL', 'InternetService_No', 'InternetService_Fiber optic']
            pm_dummy_columns = ['PaymentMethod_Electronic check', 'PaymentMethod_Bank transfer (automatic)',
                'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Mailed check']
            df_encoded = self.df
            for column_name in is_dummy_columns + pm_dummy_columns:
                df_encoded[column_name] = 0
            df_encoded['InternetService_' + df_encoded['InternetService']] = 1
            df_encoded['PaymentMethod_' + df_encoded['PaymentMethod']] = 1
            df_encoded = df_encoded.drop('InternetService', axis=1, errors='ignore')
            df_encoded = df_encoded.drop('PaymentMethod', axis=1, errors='ignore')

        # self.scale_columns()

        # Drop Customer ID - Doesn't provide numerical info, and hides personal info
        self.cleansed_df = df_encoded.drop('customerID', axis=1, errors='ignore')

    def get_scaler(self) -> TransformerMixin:
        return ColumnTransformer([ ("scale", MinMaxScaler(), self.numeric_columns) ], remainder='passthrough')

    def engineer_features(self):
        self.cleansed_df_bkup = self.cleansed_df.copy(deep=True)
        # Combine Short-Term and Fiber
        self.cleansed_df['ShortTermFiber'] = self.cleansed_df['Contract'] * self.cleansed_df['InternetService_Fiber optic']

        # Normalize charges to customer length
        scaler = MinMaxScaler()
        self.cleansed_df['AvgChargesPerMonth'] = self.cleansed_df['TotalCharges'] / (1.0 + self.cleansed_df['tenure'])
        self.cleansed_df['AvgChargesPerMonth'] = scaler.fit_transform(self.cleansed_df[['AvgChargesPerMonth']])

        # New customers (in first year)
        self.cleansed_df['Is_New_Customer'] = (self.cleansed_df['tenure'] <= 12).astype(int)
        self.data_description = 'Feature Engineered'

    def remove_low_impact_features(self):
        self.cleansed_df = self.cleansed_df_bkup.copy(deep=True)
        low_impact_features = ['DeviceProtection', 'OnlineBackup', 'PaymentMethod_Electronic check',
                      'numAdminTickets', 'TechSupport', 'Dependents', 'MonthlyCharges', 'PaymentMethod_Mailed check',
                      'MultipleLines', 'PhoneService', 'StreamingMovies', 'StreamingTV', 'PaperlessBilling',
                      'PaymentMethod_Credit card (automatic)', 'gender', 'TotalCharges', 'PaymentMethod_Bank transfer (automatic)',
                      'Partner', 'SeniorCitizen']
        self.cleansed_df =  self.cleansed_df.drop(low_impact_features, axis=1)
        for col in low_impact_features:
            if col in self.numeric_columns:
                self.numeric_columns.remove(col)
        self.data_description = 'Low Impact Features Removed'

    # Used for experiments - nead *both* Input *and* Output for comparison
    def get_input_output(self) -> ModelData:
        y = self.cleansed_df['Churn']
        X = self.cleansed_df.drop('Churn', axis=1)
        return ModelData(X, y, self.data_description)
    
    # Used for verifying predictive input *before* cleansing turns up issues.
    def get_raw(self) -> DataFrame:
        return self.df
    
    def set_raw(self, df: DataFrame):
        self.df = df
    
    # Used for predictions - output is *unknown*, so only need input
    def get_cleansed_input(self) -> DataFrame:
        return self.cleansed_df
    
    # Since Yes was translated to 1, and No to 0
    # But here give a more descriptive value
    @staticmethod
    def get_churn_translation():
        return ['Not likely to Churn', 'Likely to Churn']







    

        
