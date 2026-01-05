import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def load_raw_data(file_path='data/raw/cdc_diabetes.csv'):
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from {file_path}")
    return df


def prepare_data(df, test_size=0.2, random_state=42, scale_features=True):
    if 'Diabetes_binary' not in df.columns:
        raise ValueError("Target column 'Diabetes_binary' not found in dataset")
    
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    if scale_features:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X.columns,
            index=X_test.index
        )
        print("Features have been standardized (scaled)")
    
    return X_train, X_test, y_train, y_test


def add_interaction_terms(X):
    X = X.copy()
    
    # Important medical interactions
    # Age interactions (older + risk factor = higher risk)
    if 'Age' in X.columns and 'HvyAlcoholConsump' in X.columns:
        X['Age_x_Alcohol'] = X['Age'] * X['HvyAlcoholConsump']
    
    if 'Age' in X.columns and 'BMI' in X.columns:
        X['Age_x_BMI'] = X['Age'] * X['BMI']
    
    if 'Age' in X.columns and 'HighBP' in X.columns:
        X['Age_x_HighBP'] = X['Age'] * X['HighBP']
    
    # BMI interactions (obesity + other factors)
    if 'BMI' in X.columns and 'PhysActivity' in X.columns:
        X['BMI_x_NoActivity'] = X['BMI'] * (1 - X['PhysActivity'])
    
    if 'BMI' in X.columns and 'HighBP' in X.columns:
        X['BMI_x_HighBP'] = X['BMI'] * X['HighBP']
    
    # Lifestyle interactions
    if 'HvyAlcoholConsump' in X.columns and 'Smoker' in X.columns:
        X['Alcohol_x_Smoker'] = X['HvyAlcoholConsump'] * X['Smoker']
    
    print(f"Added {len(X.columns) - len(X.columns) + 6} interaction terms")
    
    return X


def load_and_prepare_data(file_path='data/raw/cdc_diabetes.csv', 
                          test_size=0.2, 
                          random_state=42,
                          scale_features=True,
                          add_interactions=False):
    df = load_raw_data(file_path)
    
    # Add interaction terms before splitting if requested
    if add_interactions:
        print("\n Creating interaction terms...")
        # Separate target first
        if 'Diabetes_binary' in df.columns:
            y = df['Diabetes_binary']
            X = df.drop('Diabetes_binary', axis=1)
            X = add_interaction_terms(X)
            df = pd.concat([X, y], axis=1)
    
    return prepare_data(df, test_size, random_state, scale_features)


if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...\n")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("\n✅ Data loaded and prepared successfully!")
    print(f"\nFeature columns: {list(X_train.columns)}")
