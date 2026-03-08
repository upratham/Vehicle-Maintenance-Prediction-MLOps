import os
from pyclbr import Class
import numpy as np
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")


load_dotenv(override=True)
sns.set(style='whitegrid')



class DataPreprocessor:
    def __init__(self, connection_url, db_name, collection_name):
        self.client = MongoClient(connection_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.db_name = db_name
        self.collection_name = collection_name
    
    def load_data_from_mongodb(self) -> pd.DataFrame:
        """Load data from MongoDB and return as a pandas DataFrame."""       

        self.data = list(self.collection.find())
        
        self.df = pd.DataFrame(self.data)
        
        # Drop MongoDB's default _id column
        if "_id" in self.df.columns:
            self.df.drop(columns=["_id"], inplace=True)
        
        self.client.close()
        
        print(f"Loaded {len(self.df)} records from {self.db_name}.{self.collection_name}")
        
        return self.df
    
    def data_preprocessing(self,df,REFERENCE_DATE,plot_dir) -> tuple:
        # checking for duplicates and missing values
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        reference_date = REFERENCE_DATE

        for col in ["Last_Service_Date", "Warranty_Expiry_Date"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col + "_days"] = (reference_date - df[col]).dt.days

        df.drop(columns=["Last_Service_Date", "Warranty_Expiry_Date"], inplace=True)
        # Cap outliers in all numerical columns (except target) using IQR method to clip extreme values within acceptable bounds

        numerical_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols_raw.remove("Need_Maintenance")

        for col in numerical_cols_raw:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)
        # Encode categorical columns, compute Spearman correlation heatmap, calculate mutual information scores,
        # and drop features with zero predictive power toward the target variable

        df_encoded = df.copy()
        cat_cols_tmp = df_encoded.select_dtypes(include="object").columns.tolist()
        le = LabelEncoder()
        for c in cat_cols_tmp:
            df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))

        spearman_corr = df_encoded.corr(method="spearman")

        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(
            spearman_corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            linewidths=0.5,
            annot_kws={"size": 7},
        )
        plt.title("Spearman Rank-Correlation Heat-map", fontsize=14, pad=12)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "spearman_heatmap.png"), dpi=300)
        #plt.show()

        X_tmp = df_encoded.drop(columns=["Need_Maintenance"])
        y_tmp = df_encoded["Need_Maintenance"]

        mi_scores = mutual_info_classif(X_tmp, y_tmp, discrete_features="auto", random_state=42)
        mi_series = pd.Series(mi_scores, index=X_tmp.columns).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        mi_series.plot(kind="bar", color="steelblue")
        plt.title("Mutual Information Scores vs. Need_Maintenance")
        plt.ylabel("MI Score")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "mi_scores.png"), dpi=300)
        #plt.show()

        zero_mi_features = mi_series[mi_series == 0].index.tolist()
        if zero_mi_features:
            df.drop(columns=zero_mi_features, inplace=True)

        # Define ordinal features with their ranked category orders (low → high)
        ordinal_features = {
            "Maintenance_History": ["Poor", "Average", "Good"],
            "Tire_Condition":      ["Worn Out", "Good", "New"],
            "Brake_Condition":     ["Worn Out", "Good", "New"],
            "Battery_Status":      ["Weak", "Good", "Strong"],
        }

        # Define nominal features (no natural order) that exist in the dataframe
        nominal_features = [
            c for c in ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]
            if c in df.columns
        ]

        # Select all numerical columns excluding the target variable
        numerical_features = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != "Need_Maintenance"
        ]

        # Keep only ordinal features that are present in the dataframe
        ordinal_features = {k: v for k, v in ordinal_features.items() if k in df.columns}

        # Separate features (X) and target variable (y)
        X = df.drop(columns=["Need_Maintenance"])
        y = df["Need_Maintenance"]

        # Split data into 80% train and 20% test, stratified to preserve class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
    
        ordinal_transformer = OrdinalEncoder(
            categories=[ordinal_features[k] for k in ordinal_features],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )

        nominal_transformer = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        numerical_transformer = RobustScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ("ord", ordinal_transformer,  list(ordinal_features.keys())),
                ("nom", nominal_transformer,  nominal_features),
                ("num", numerical_transformer, numerical_features),
            ],
            remainder="drop",
        )

        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc  = preprocessor.transform(X_test)
        # Handle class imbalance with SMOTE
      
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_proc, y_train)

        return X_train_sm, X_test_proc, y_train_sm, y_test

def main():
   #Constants
    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)
    DBNAME="605_Project_Data"
    CONNECTION_URL=os.getenv("CONNECTION_URL")
    COLLECTION_NAME=os.getenv("COLLECTION_NAME")
    REFERENCE_DATE = pd.Timestamp("2026-03-07")
    pre_processor = DataPreprocessor(CONNECTION_URL, DBNAME, COLLECTION_NAME)
    df= pre_processor.load_data_from_mongodb()
    X_train_sm, X_test_proc, y_train_sm, y_test= pre_processor.data_preprocessing(df,REFERENCE_DATE,PLOT_DIR)
    return X_train_sm, X_test_proc, y_train_sm, y_test

if __name__ == "__main__":
    X_train_sm, X_test_proc, y_train_sm, y_test = main()