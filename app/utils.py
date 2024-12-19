import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(data_path):
    """
    Preprocess the Mental Health Care dataset and return the preprocessed DataFrame.
    """
    # Read the CSV file
    data = pd.read_csv(data_path)

    # Select required columns
    features = ['Indicator', 'Group', 'Subgroup', 'Phase', 'Time Period Start Date', 'Time Period End Date']
    target_value = 'Value'  # Target for scaling dummies
    data_subset = data[features + [target_value]].copy()

    # Extract year and month, and drop unused columns
    data_subset['Start Year'] = pd.to_datetime(data_subset['Time Period Start Date']).dt.year
    data_subset['Start Month'] = pd.to_datetime(data_subset['Time Period Start Date']).dt.month
    data_subset['End Year'] = pd.to_datetime(data_subset['Time Period End Date']).dt.year
    data_subset['End Month'] = pd.to_datetime(data_subset['Time Period End Date']).dt.month
    data_subset = data_subset.drop(['Time Period Start Date', 'Time Period End Date'], axis=1)

    # Drop NaN values
    data_subset = data_subset.dropna()

    # Add Group_Subgroup column and process ranges
    data_subset['Group_Subgroup'] = data_subset['Group'] + "_" + data_subset['Subgroup']
    
    import re
    def range_to_midpoint(subgroup):
        match = re.match(r"(\d+)\s*-\s*(\d+)", subgroup)
        if match:
            lower, upper = map(int, match.groups())
            return (lower + upper) / 2
        return subgroup

    data_subset['Subgroup_processed'] = data_subset['Subgroup'].apply(range_to_midpoint)

    # One-hot encode Group and Group_Subgroup
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_features = encoder.fit_transform(data_subset[['Group', 'Group_Subgroup']])
    encoded_features = pd.DataFrame(
        encoded_features.toarray(), 
        columns=encoder.get_feature_names_out(['Group', 'Group_Subgroup'])
    )

    # Concatenate encoded features with original data
    data_encoded = pd.concat([data_subset.reset_index(drop=True), encoded_features], axis=1)

    return data_encoded