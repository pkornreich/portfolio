# BS"D

import pandas as pd
from pandas import DataFrame, Series
from sklearn.preprocessing import MinMaxScaler

from constants import Constants

class DataService:

    def __init__(self):
        self.df = DataService.read_data()
        self.cleansed_df = None

    @staticmethod
    def read_data() -> DataFrame:
        full_path = Constants.DATA_PATH + Constants.DATA_FILE_NAME
        df: DataFrame = pd.read_excel(full_path)
        return df
    
    def convert_binary_values(self):
        # Gender
        self.df['gender'] = self.df['gender'].replace({'Female': 1, 'Male': 0})
        # Simple Yes/No columns
        simple_binary_columns = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        for column in simple_binary_columns:
            self.df[column] = self.df[column].replace({'Yes': 1, 'No': 0})
        # Phone Service
        self.df['MultipleLines'] = self.df['MultipleLines'].replace({'Yes': 1, 'No': 0, 'No phone service': 0})
        # Internet Service
        internet_binary_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                   'TechSupport', 'StreamingTV', 'StreamingMovies']
        for column in internet_binary_columns:
            self.df[column] = self.df[column].replace({'Yes': 1, 'No': 0, 'No internet service': 0})

    
    def cleanse_data(self):
        # Change appropriate string columns to numeric
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')

        # Handle Missing Values
        null_count = self.df['TotalCharges'].isna().sum()
        non_null_count = self.df['TotalCharges'].count()
        message = f" Percentage of Null Total Charges Values: {round(null_count*100/(non_null_count + null_count), 2)}%"
        # Shows non-null count is < 1%
        print(message) 
        # So, can drop them
        self.df.dropna(subset=['TotalCharges'], inplace=True)

        # Binary Value Columns
        self.convert_binary_values()
        # Convert naturally ordered String columns to numeric
        self.df['Contract'] = self.df['Contract'].replace({'Month-to-month': 0,  'One year': 1, 'Two year': 2})

        # Multi-value string columns that don't have a natural order - One-Hot Encode
        df_encoded = pd.get_dummies(self.df, columns=['InternetService', 'PaymentMethod'])

        # Scale Appropriate Columns
        scaler = MinMaxScaler()
        df_encoded['tenure'] = scaler.fit_transform(df_encoded[['tenure']])
        df_encoded['MonthlyCharges'] = scaler.fit_transform(df_encoded[['MonthlyCharges']])
        df_encoded['TotalCharges'] = scaler.fit_transform(df_encoded[['TotalCharges']])
        df_encoded['numAdminTickets'] = scaler.fit_transform(df_encoded[['numAdminTickets']])
        df_encoded['numTechTickets'] = scaler.fit_transform(df_encoded[['numTechTickets']])

        # Drop Customer ID - Doesn't provide numerical info, and hides personal info
        self.cleansed_df = df_encoded.drop('customerID', axis=1)

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

    def remove_low_impact_features(self):
        self.cleansed_df = self.cleansed_df_bkup.copy(deep=True)
        self.cleansed_df =  self.cleansed_df.drop(['DeviceProtection', 'OnlineBackup', 'PaymentMethod_Electronic check',
                      'numAdminTickets', 'TechSupport', 'Dependents', 'MonthlyCharges', 'PaymentMethod_Mailed check',
                      'MultipleLines', 'PhoneService', 'StreamingMovies', 'StreamingTV', 'PaperlessBilling',
                      'PaymentMethod_Credit card (automatic)', 'gender', 'TotalCharges', 'PaymentMethod_Bank transfer (automatic)',
                      'Partner', 'SeniorCitizen'], axis=1)

    def get_input_output(self) -> tuple[DataFrame, Series]:
        y = self.cleansed_df['Churn']
        X = self.cleansed_df.drop('Churn', axis=1)
        return X, y








    

        
