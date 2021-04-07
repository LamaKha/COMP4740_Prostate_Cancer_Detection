import pandas as pd

def load_and_process_data(file_name):
    data = pd.read_csv(file_name)

    # Set index to ID values (otherwise keys are wrong after transposition)
    data = data.set_index('ID')

    # Transpose the data (because features are rows in CSV and they should be columns)
    data = data.transpose()

    # Remove the features that only contain 0 values
    data = data.loc[:, (data != 0).any(axis=0)]

    # Add new target variable (GRADE_GROUP)
    targets = []
    for index, row in data.iterrows():
        # GLEASON SCORE = 6 -> GRADE GROUP = 1
        if (row['GLEASON_SCORE'] == 6):
            targets.append(1)
        # GLEASON SCORE = 7 -> GRADE GROUP = 2/3
        elif (row['GLEASON_SCORE'] == 7):
            # PRIMARY SCORE = 3 -> GRADE GROUP = 2
            if (row['GLEASON_PATTERN_PRIMARY'] == 3):
                targets.append(2)
            # PRIMARY SCORE = 4 -> GRADE GROUP = 3
            else:
                targets.append(3)
        # GLEASON SCORE = 8 -> GRADE GROUP = 4
        elif (row['GLEASON_SCORE'] == 8):
            targets.append(4)
        # GLEASON SCORE = 9/10 -> GRADE GROUP = 5
        else:
            targets.append(5)
    data.insert(len(data.columns), "GRADE_GROUP", targets)

    # Identify independant and dependant variables
    # Remove the last 6 columns
    X = data.iloc[:, : -6]
    # Temporarily setting target variable to 'GLEASON_PATTERN_PRIMARY'
    y = data['GLEASON_SCORE']

    return X, y