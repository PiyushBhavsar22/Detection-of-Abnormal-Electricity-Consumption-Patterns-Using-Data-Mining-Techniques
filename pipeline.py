import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
import os
import glob
warnings.filterwarnings('ignore')

def run_pipeline():
    print("==========================================")
    print("STARTING ADVANCED ML PIPELINE")
    print("==========================================")

    # ==========================================
    # PHASE 1: LOAD ALL DATASETS 
    # ==========================================
    print("\n--- PHASE 1: LOADING DATA ---")
    
    block_folder = r"E:/Project/Test Andrew Project/data/hhblock_dataset"
    block_files = glob.glob(os.path.join(block_folder, "**", "*.csv"), recursive=True)
    
    print(f"DEBUG: Searching inside -> {block_folder}")
    print(f"DEBUG: Found {len(block_files)} CSV files anywhere inside this directory.")
    
    if not block_files:
        raise FileNotFoundError(f"No CSV files found in '{block_folder}'. Check folder name!")

    print(f"Loading {len(block_files)} block files... (This might take a few minutes!)")
    df_list = [pd.read_csv(file) for file in block_files]
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df):,} rows of smart meter data.")
    
    df_house = pd.read_csv(r"E:/Project/Test Andrew Project/data/informations_households.csv", encoding='latin-1')
    df_weather_daily = pd.read_csv(r"E:/Project/Test Andrew Project/data/weather_daily_darksky.csv", encoding='latin-1')
    df_holidays = pd.read_csv(r"E:/Project/Test Andrew Project/data/uk_bank_holidays.csv", encoding='latin-1')
    df_weather_hourly = pd.read_csv(r"E:/Project/Test Andrew Project/data/weather_hourly_darksky.csv", encoding='latin-1')
    df_acorn = pd.read_csv(r"E:/Project/Test Andrew Project/data/acorn_details.csv", encoding='latin-1')

    # ==========================================
    # PHASE 2: ADVANCED PREPROCESSING & MERGING
    # ==========================================
    print("\n--- PHASE 2: PREPROCESSING & MERGING ---")
    
    df.columns = df.columns.str.strip()
    df_house.columns = df_house.columns.str.strip()
    
    df_weather_hourly['day'] = pd.to_datetime(df_weather_hourly['time']).dt.normalize()
    hourly_agg = df_weather_hourly.groupby('day').agg(
        temp_hr_std=('temperature', 'std'), 
        humidity_hr_mean=('humidity', 'mean')
    ).reset_index()
    
    df_weather_daily['day'] = pd.to_datetime(df_weather_daily['time']).dt.normalize()
    df_weather_master = pd.merge(df_weather_daily, hourly_agg, on='day', how='left')
    
    df_acorn['feature_name'] = df_acorn['MAIN CATEGORIES'].astype(str) + "_" + df_acorn['REFERENCE'].astype(str)
    df_acorn = df_acorn.drop_duplicates(subset=['feature_name'])
    
    df_acorn_t = df_acorn.set_index('feature_name').drop(columns=['MAIN CATEGORIES', 'CATEGORIES', 'REFERENCE']).T
    df_acorn_t.index.name = 'Acorn'
    df_acorn_t = df_acorn_t.reset_index()
    
    for col in df_acorn_t.columns:
        if col != 'Acorn':
            df_acorn_t[col] = pd.to_numeric(df_acorn_t[col], errors='coerce').fillna(0)
    
    df_house_enriched = pd.merge(df_house, df_acorn_t, on='Acorn', how='left')
    
    cols_to_encode = [col for col in ['Acorn', 'stdorToU'] if col in df_house_enriched.columns]
    df_house_encoded = pd.get_dummies(df_house_enriched, columns=cols_to_encode, drop_first=True)

    holiday_time_col = 'Bank holidays' if 'Bank holidays' in df_holidays.columns else df_holidays.columns[0]
    df_holidays['day'] = pd.to_datetime(df_holidays[holiday_time_col], errors='coerce').dt.normalize()
    df_holidays['is_holiday'] = 1

    if 'tpep_pickup_datetime' in df.columns:
        date_col = 'tpep_pickup_datetime'
    elif 'day' in df.columns:
        date_col = 'day'
    else:
        date_col = df.columns[1] 

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df['day_formatted'] = df[date_col].dt.normalize()
    df['energy_kwh'] = pd.to_numeric(df['energy_kwh'] if 'energy_kwh' in df.columns else df.iloc[:, 2], errors='coerce').fillna(0)

    # ==========================================
    # PHASE 3: FEATURE ENGINEERING
    # ==========================================
    print("\n--- PHASE 3: FEATURE ENGINEERING ---")
    df['hour'] = df[date_col].dt.hour
    df['is_peak'] = df['hour'].apply(lambda x: 1 if 17 <= x <= 21 else 0)

    daily_features = df.groupby(['LCLid', 'day_formatted']).agg(
        total_daily_kwh=('energy_kwh', 'sum'),
        daily_variance=('energy_kwh', 'var'),
        peak_sum=('energy_kwh', lambda x: x[df.loc[x.index, 'is_peak'] == 1].sum()),
        off_peak_sum=('energy_kwh', lambda x: x[df.loc[x.index, 'is_peak'] == 0].sum())
    ).reset_index()
    
    daily_features.rename(columns={'day_formatted': 'day'}, inplace=True)

    daily_features['daily_variance'] = daily_features['daily_variance'].fillna(0)
    daily_features['peak_to_offpeak_ratio'] = daily_features['peak_sum'] / (daily_features['off_peak_sum'] + 0.001)

    daily_features = pd.merge(daily_features, df_weather_master[['day', 'temperatureMax', 'temp_hr_std', 'cloudCover']], on='day', how='left')
    daily_features = pd.merge(daily_features, df_holidays[['day', 'is_holiday']], on='day', how='left')
    
    daily_features['is_holiday'] = daily_features['is_holiday'].fillna(0).astype(int)
    daily_features.fillna(0, inplace=True) 

    # ==========================================
    # PHASE 4: UNSUPERVISED ANOMALY DETECTION
    # ==========================================
    print("\n--- PHASE 4: UNSUPERVISED LEARNING (Isolation Forest) ---")
    ml_features = ['total_daily_kwh', 'daily_variance', 'peak_sum', 'off_peak_sum', 
                   'peak_to_offpeak_ratio', 'temperatureMax', 'temp_hr_std', 'is_holiday']
    
    X_unsupervised = daily_features[ml_features]
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    daily_features['anomaly_score'] = iso_forest.fit_predict(X_unsupervised)
    daily_features['is_anomaly'] = daily_features['anomaly_score'].apply(lambda x: 1 if x == -1 else 0)

    # ==========================================
    # PHASE 5: SUPERVISED TRAINING SETUP (THE FIX)
    # ==========================================

    print("\n--- PHASE 5: SUPERVISED TRAINING SETUP ---")
    
    # Group data by user
    agg_dict = {col: 'mean' for col in ml_features}
    agg_dict['is_anomaly'] = 'sum' 
    user_profiles = daily_features.groupby('LCLid').agg(agg_dict).reset_index()

    # Merge demographics, fill missing with 0
    user_profiles = pd.merge(user_profiles, df_house_encoded, on='LCLid', how='left')
    user_profiles.fillna(0, inplace=True) 

    # Label the top 5% anomalies as thieves
    anomaly_threshold = user_profiles['is_anomaly'].quantile(0.95)
    user_profiles['is_confirmed_thief'] = (user_profiles['is_anomaly'] >= anomaly_threshold).astype(int)

    # STRICTLY isolate the 8 behavioral features so the AI doesn't get confused by demographics
    X_supervised = user_profiles[ml_features]
    y_supervised = user_profiles['is_confirmed_thief']

    X_supervised.rename(columns=str, inplace=True)
    X_supervised.columns = pd.Index([str(c) for c in X_supervised.columns])

    # ==========================================
    # PHASE 6: SMOTE, EVALUATION & EXPORT
    # ==========================================
    print("\n--- PHASE 6: RIGOROUS EVALUATION & SMOTE ---")

    # 1. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_supervised, y_supervised, test_size=0.2, random_state=42, stratify=y_supervised
    )

    # 2. Export the untouched 20% test data for the Jupyter Notebook visualizations
    X_test.to_csv("X_test_sample.csv", index=False)
    y_test.to_csv("y_test_sample.csv", index=False)
    print("Exported unseen test data to CSV for Jupyter Notebook visualizations.")

    # 3. Apply SMOTE strictly to the Training Data to fix Class Imbalance
    print(f"\nOriginal Training Class Balance: \n{y_train.value_counts()}")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"New SMOTE Training Class Balance: \n{y_train_smote.value_counts()}\n")

    # 4. Define the models for the head-to-head showdown
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
        "Gradient Boosting (Complex)": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Random Forest (Proposed Model)": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }

    # 5. Train on SMOTE data and Evaluate on UNTOUCHED Test data
    for name, model in models.items():
        print(f"[ Evaluating: {name} ]")
        model.fit(X_train_smote, y_train_smote) 
        y_pred = model.predict(X_test)
        
        print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("-" * 50)

    # 6. Final API Deployment Prep: Apply SMOTE to ALL data, then train
    print("\nRetraining Proposed Model on 100% of SMOTE data for API deployment...")
    X_full_smote, y_full_smote = smote.fit_resample(X_supervised, y_supervised)
    
    final_model = models["Random Forest (Proposed Model)"]
    final_model.fit(X_full_smote, y_full_smote)

    # Save the final AI brain
    joblib.dump(final_model, "theft_detection_model.pkl")
    joblib.dump(list(X_supervised.columns), "model_features.pkl")

    print("\nSUCCESS: SMOTE Pipeline Complete! Model saved for API.")

if __name__ == "__main__":
    run_pipeline()