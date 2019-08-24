import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

#CERVICAL
def cervical():
    data = pd.read_csv('cdata.csv')

    df_full = pd.read_csv('cdata.csv')

    df_fullna = df_full.replace('?', np.nan)

    df = df_fullna  # making temporary save

    df = df.convert_objects(convert_numeric=True)  # turn data into numeric type for computation

    df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
    df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
    df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
    df['Smokes'] = df['Smokes'].fillna(1)
    df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
    df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
    df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
    df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(
        df['Hormonal Contraceptives (years)'].median())
    df['IUD'] = df['IUD'].fillna(0)  # Under suggestion
    df['IUD (years)'] = df['IUD (years)'].fillna(0)  # Under suggestion
    df['STDs'] = df['STDs'].fillna(1)
    df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
    df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(df['STDs:condylomatosis'].median())
    df['STDs:cervical condylomatosis'] = df['STDs:cervical condylomatosis'].fillna(
        df['STDs:cervical condylomatosis'].median())
    df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(
        df['STDs:vaginal condylomatosis'].median())
    df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(
        df['STDs:vulvo-perineal condylomatosis'].median())
    df['STDs:syphilis'] = df['STDs:syphilis'].fillna(df['STDs:syphilis'].median())
    df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(
        df['STDs:pelvic inflammatory disease'].median())
    df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(df['STDs:genital herpes'].median())
    df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(
        df['STDs:molluscum contagiosum'].median())
    df['STDs:AIDS'] = df['STDs:AIDS'].fillna(df['STDs:AIDS'].median())
    df['STDs:HIV'] = df['STDs:HIV'].fillna(df['STDs:HIV'].median())
    df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(df['STDs:Hepatitis B'].median())
    df['STDs:HPV'] = df['STDs:HPV'].fillna(df['STDs:HPV'].median())
    df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(
        df['STDs: Time since first diagnosis'].median())
    df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(
        df['STDs: Time since last diagnosis'].median())

    df = pd.get_dummies(data=df,
                        columns=['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV',
                                 'Dx'])

    # print(df.isnull().sum())

    # print(data.head())

    df.drop(['Hinselmann', 'Schiller', 'Citology', 'Biopsy'], axis=1, inplace=True)

    # print(data.isnull().sum())

    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from sklearn import metrics

    predictors = df.columns[0:]
    target = 'Hinselmann'

    # print(df.isnull().sum())

    X = df.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])

    # Split the dataset in train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print ('training set : %i , test set : %i' % (X_train.shape[0], X_test.shape[0]))

    # Importing the model:
    from sklearn import svm

    # Initiating the model:
    svm = svm.SVC(kernel='linear', gamma='auto')

    scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=10).mean()

    print("The mean accuracy for Hinselmann using SVM is %s" % round(scores * 100, 2))

    target = 'Schiller'

    # print(df.isnull().sum())

    X = df.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])

    # Split the dataset in train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print ('training set : %i , test set : %i' % (X_train.shape[0], X_test.shape[0]))

    # Importing the model:
    from sklearn import svm

    # Initiating the model:
    svm = svm.SVC(kernel='linear', gamma='auto')

    scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=10).mean()

    print("The mean accuracy for Schiller using SVM is %s" % round(scores * 100, 2))

    target = 'Citology'

    # print(df.isnull().sum())

    X = df.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])

    # Split the dataset in train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print ('training set : %i , test set : %i' % (X_train.shape[0], X_test.shape[0]))

    # Importing the model:
    from sklearn import svm

    # Initiating the model:
    svm = svm.SVC(kernel='linear', gamma='auto')

    scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=10).mean()

    print("The mean accuracy for Citology using SVM is %s" % round(scores * 100, 2))

    target = 'Biopsy'

    # print(df.isnull().sum())

    X = df.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])

    # Split the dataset in train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print ('training set : %i , test set : %i' % (X_train.shape[0], X_test.shape[0]))

    # Importing the model:
    from sklearn import svm

    # Initiating the model:
    svm = svm.SVC(kernel='linear', gamma='auto')

    scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=10).mean()

    print("The mean accuracy for Biopsy using SVM is %s" % round(scores * 100, 2))


#LUNG
def lung():
    def init():

        df = pd.read_csv("ldata.csv")

        # Dropping the 'id' column from the dataset
        # df=df.drop("id",1)
        # df=df.drop("Unnamed: 32",1)

        # Mapping M to 1 and B to 0 in the output Label DataFrame
        # df['status']=df['status'].map({'R':1,'N':0})

        # Split Data into training and test (70% and 30%)
        train, test = train_test_split(df, test_size=0.3, random_state=1)

        # Training Data
        train_x = train.loc[:, 'Age': 'Alcohol']
        train_y = train.loc[:, ['Result']]

        # Testing Data
        test_x = test.loc[:, 'Age': 'Alcohol']
        test_y = test.loc[:, ['Result']]

        # Converting Traing and Test Data to numpy array
        train_x = np.asarray(train_x)
        train_y = np.asarray(train_y)
        test_x = np.asarray(test_x)
        test_y = np.asarray(test_y)

        print(train_x)

        # Calling the model function to train a Logistic Regression Model on Training Data
        d = model(train_x.T, train_y.T, num_of_iterations=10000, alpha=0.0001)

        costs = d["costs"]
        w = d["w"]
        b = d["b"]

        # Drawing the plot between cost and number of iterations
        #plt.plot(costs)
        #plt.title("Cost vs #Iterations")
        #plt.xlabel("Number of Iterations ( * 10)")
        #plt.ylabel("Cost")

        # Now, calculating the accuracy on Training and Test Data
        Y_prediction_train = predict(train_x.T, w, b)
        Y_prediction_test = predict(test_x.T, w, b)

        trial = [[50, 8, 7, 7]]
        abc = np.asarray(trial)
        ytry = predict(abc.T, w, b)
        #print(ytry)

        #print("\nTrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y.T)) * 100))

        #print("\nTest accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y.T)) * 100))

        #print(train_x)

        return ytry[0]

    # Function to initialize the weights and bias
    def initialize(m):

        w = np.zeros((m, 1))
        b = 0

        return w, b

    # Function to calculate sigmoid of x
    def sigmoid(X):
        return 1 / (1 + np.exp(- X))

        # Function for doing forward and back propogation

    def propogate(X, Y, w, b):

        m = X.shape[1]  # Number of training examples

        # Forward Propogation, calculating the cost
        Z = np.dot(w.T, X) + b;
        A = sigmoid(Z)
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

        # Back Propogation , calculating the gradients
        dw = (1 / m) * np.dot(X, (A - Y).T)
        db = (1 / m) * np.sum(A - Y)

        grads = {"dw": dw, "db": db}

        return grads, cost

    # Function for performing Grdient Descent
    def optimize(X, Y, w, b, num_of_iterations, alpha):

        costs = []

        for i in range(num_of_iterations):

            grads, cost = propogate(X, Y, w, b)

            dw = grads["dw"]
            db = grads["db"]

            w = w - alpha * dw
            b = b - alpha * db

            # Storing tthe cost at interval of every 10 iterations
            if i % 10 == 0:
                costs.append(cost)
                print("cost after %i iteration : %f" % (i, cost))

        parameters = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

        return parameters, grads, costs

    # Function for doing the predictions on the data set (mapping probabilities to 0 or 1)
    def predict(X, w, b):

        m = X.shape[1]  # Number of training examples

        y_prediction = np.zeros((1, m))

        w = w.reshape(X.shape[0], 1)

        A = sigmoid(np.dot(w.T, X) + b)

        for i in range(A.shape[1]):

            if (A[0, i] < 0.5):
                y_prediction[0, i] = 0
            else:
                y_prediction[0, i] = 1

        return y_prediction

    # Function for calculating the Logistic Regression Model
    def model(Xtrain, Ytrain, num_of_iterations, alpha):

        dim = Xtrain.shape[0]  # Number of features

        w, b = initialize(dim)

        parameters, grads, costs = optimize(Xtrain, Ytrain, w, b, num_of_iterations, alpha)

        w = parameters["w"]
        b = parameters["b"]

        d = {"w": w, "b": b, "costs": costs}

        return d

    # Calling the init function to start the program
    x=init()
    return x

#DIAGNOSIS
def bdiag():
    data = pd.read_csv('ddata.csv')

    data.drop('id', axis=1, inplace=True)
    data.drop('Unnamed: 32', axis=1, inplace=True)

    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    datas = pd.DataFrame(preprocessing.scale(data.iloc[:, 1:32]))
    datas.columns = list(data.iloc[:, 1:32].columns)
    datas['diagnosis'] = data['diagnosis']

    data_mean = data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                      'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
                      'fractal_dimension_mean']]

    # Splitting data:

    from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
    from sklearn import metrics

    predictors = data_mean.columns[1:]
    target = "diagnosis"

    X = data_mean.loc[:, predictors]
    y = np.ravel(data.loc[:, [target]])

    # Split the dataset in train and test:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print ('training set : %i , test set : %i' % (X_train.shape[0], X_test.shape[0]))

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # SVM
    from sklearn import svm
    svm = svm.SVC(kernel='rbf', gamma='auto')
    scores = cross_val_score(svm, X_train, y_train, scoring='accuracy', cv=10).mean()
    print("The mean accuracy using SVM is %s" % round(scores * 100, 2))

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    scores = cross_val_score(rf, X_train, y_train, scoring='accuracy', cv=10).mean()
    print("The mean accuracy using Random Forest is %s" % round(scores * 100, 2))

    # Trial
    print("\nAnswer:")
    trial = [[13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766]]
    svm.fit(X_train, y_train)
    ans=svm.predict(trial)
    an=ans[0]
    return an


def bprog():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    bankdata = pd.read_csv("pda2.csv")
    trial = pd.read_csv("check.csv")
    #trial=tria.drop(tria.columns[[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]], axis=1, )

    print bankdata.head()

    # a=bankdata.drop(bankdata.columns[[11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]], axis=1, )
    # print a.head()
    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.svm import SVC
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)



    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_try = svclassifier.predict(trial)
    return y_try[0]

def main():


    when=raw_input("Are you currently suffering from breast cancer?(Assuming that you are either having BC or have had it in the past): ")
    if when=="yes":
        print("enter details to check if its m or b(auto entered)")
            #trial = [[13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766]]

        ans=bdiag()
        if ans==0:
            print("The cancer is benign")

        else:
            print ("The cancer is malignant")
                #p=p+3000
    else:
        print("Enter trial values to check if it will recurr(auto entered)")
        ans1=bprog()
        if ans1==0:
            print("The cancer will not recur")

        else:
            print ("The cancer will recur")
                #p=p+4000

main()