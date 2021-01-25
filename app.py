try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    import numpy as np
    import pandas as pd
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
    from tkinter import simpledialog
except ImportError:
    print('Please install the required modules')


root = tk.Tk()


def showUI():
    file_path = OpenFileExplorer()
    x_headers, y_header, encoding = GetUserData()

    return file_path, x_headers, y_header, encoding


def OpenFileExplorer():
    rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='/',
        initialfile='tmp',
        filetypes=[(".csv", "*.csv")])

    try:
        return rep[0]
    except:
        return ""


def GetUserData():
    headerStr = simpledialog.askstring(
        title="Features", prompt="Enter the headers (separated by commas) you wish do include as features").strip()
    HEADERS = list(map(str.strip, headerStr.split(",")))

    PRED = [simpledialog.askstring(
        title="Prediction", prompt="What is your dependant variable?").strip()]

    ENCODING = simpledialog.askstring(
        title="Encoding", prompt="Do any columns contain catigorical data? (Enter headers separated by commas)").strip()
    return HEADERS, PRED, ENCODING.split(",")


def UserPreferences(path, COLS, DEP_VARIABLE, ENCODING):
    df = pd.read_csv(path)
    for col in COLS:
        assert col in df.columns.values, "One of your headers is not in the dataset, please check for spelling and capitalization"
    assert DEP_VARIABLE[0] not in COLS, "You may have included your dependant variables in your features"

    X = df[COLS]
    y = df[DEP_VARIABLE]

    HEADERS = X.columns.values
    encoding_cols = []
    for i in range(len(HEADERS)):
        if HEADERS[i] in ENCODING:
            encoding_cols += [i]

    return X.values, y.values, encoding_cols, HEADERS


# initializing vars
X, y, ENCODING_COLS, HEADERS = UserPreferences(*showUI())
encoding_needed = False
scaling_needed = False

if len(ENCODING_COLS) > 0:
    encoding_needed = True

if len(HEADERS) <= 1:
    scaling_needed = True

# One hot tuple encoding
if encoding_needed:
    ct = ColumnTransformer(transformers=[(
        "transformer", OneHotEncoder(), ENCODING_COLS)], remainder="passthrough")
    X = np.array(ct.fit_transform(X))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)  # splitting the training set and test set

# Normalizing the dataset if there is only one feature of the data
if scaling_needed:
    std = StandardScaler()

    X_train = std.fit_transform(X_train, y_train)
    X_test = std.transform(X_test)


# Creating the model and fitting with dataset
model = LinearRegression()
model.fit(X_train, y_train)

# predicted vector
y_pred = model.predict(X_test)


y_pred = y_pred.ravel()
y_test = y_test.ravel()

try:
    with open("report.txt", "x") as f:
        pass
except FileExistsError:
    pass


def make_prediction(model, headers, scale=False, encoding=False):
    prediction = [[]]
    args = []
    for header in headers:
        value = simpledialog.askstring(title=str(
            header), prompt="Please type in the following feature: " + str(header))

        try:
            value = float(value)
        except Exception:
            pass

        prediction[0].append(value)
        args.append(header + ": " + str(value))

    prediction = np.array(prediction, dtype=object)
    if encoding:
        prediction = ct.transform(prediction)
    if scale:
        prediction = std.transform(prediction)

    with open("report.txt", "a") as report:
        report.write("\n")
        report.write(str(args) + ": " +
                     str(model.predict(prediction).flatten()))


make_prediction(model, HEADERS, scale=scaling_needed,
                encoding=encoding_needed)
