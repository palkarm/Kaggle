import datetime

import pandas as pd
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score

# Dataset Path
testDS_path = "tcd ml 2019-20 income prediction test (without labels).csv"
trainDS_path = "tcd ml 2019-20 income prediction training (with labels).csv"

def ManagingNulls(dataFrame):
    
    #Year of Record [dataType = float64] -> current year
    currentYear = float(datetime.datetime.now().year)
    dataFrame['Year of Record'] = dataFrame['Year of Record'].fillna(currentYear)

    #Gender [dataType = object] -> Unknown Gender
    dataFrame['Gender'] = dataFrame['Gender'].fillna('Unknown Gender')

    #Age [dataType = float64] -> mean
    dataFrame['Age'] = dataFrame['Age'].fillna(dataFrame['Age'].mean())

    #Profession [dataType = object] -> No Profession
    dataFrame['Profession'] = dataFrame['Profession'].fillna('No Profession')

    #University Degree [dataType = object] -> No Degree
    dataFrame['University Degree'] = dataFrame['University Degree'].fillna('No Degree')

    #Hair Color [dataType = object] -> No Hair 
    dataFrame['Hair Color'] = dataFrame['Hair Color'].fillna('No Hair')

    return dataFrame

def FormattingColumn(dataFrame):
    
    #Gender => ['0','unknown'] -> Unknown Gender | ['other'] -> Other Gender
    dataFrame['Gender'] = dataFrame['Gender'].replace(['0','unknown'],'Unknown Gender')
    dataFrame['Gender'] = dataFrame['Gender'].replace(['other'],'Other Gender')
    
    #University Degree => ['No','0'] -> No Degree
    dataFrame['University Degree'] = dataFrame['University Degree'].replace(['No','0'],'No Degree')

    #Hair Color => ['Unknown','0'] -> Unknown Hair Color
    dataFrame['Hair Color'] = dataFrame['Hair Color'].replace(['Unknown','0'],'Unknown Hair Color')

    return dataFrame

def FeatureExtraction(dataFrame):

    #create a label binary format
    lbFormat = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)
    
    #Extract Genders and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(lbFormat.fit_transform(dataFrame['Gender']),columns=lbFormat.classes_,index=dataFrame.index))

    #Remove the Gender Column
    dataFrame = dataFrame.drop(['Gender'], axis = 1)

    #Extract University Degree and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(lbFormat.fit_transform(dataFrame['University Degree']),columns=lbFormat.classes_,index=dataFrame.index))

    #Remove the University Degree Column
    dataFrame = dataFrame.drop(['University Degree'], axis = 1)

    #Extract Hair Color and join the data frame
    dataFrame = dataFrame.join(pd.DataFrame(lbFormat.fit_transform(dataFrame['Hair Color']),columns=lbFormat.classes_,index=dataFrame.index))

    #Remove the Hair Color Column
    dataFrame = dataFrame.drop(['Hair Color'], axis = 1)

    return dataFrame

def Preprocessing(dataFrame):
    
    #Managing nulls
    dataFrame = ManagingNulls(dataFrame)

    #formatting columns
    dataFrame = FormattingColumn(dataFrame)

    #feature extraction
    dataFrame = FeatureExtraction(dataFrame)

    #Initial attempt drop -> Year of record, country and city size
    #dataFrame = dataFrame.drop(['Year of Record'], axis = 1)
    dataFrame = dataFrame.drop(['Country'], axis = 1)
    dataFrame = dataFrame.drop(['Size of City'], axis = 1)

    return dataFrame

def PreprocessingTrainingDS():
    #load data
    trainingFrame = pd.read_csv(trainDS_path)

    #preprocessing - basic
    processedTrainingFrame = Preprocessing(trainingFrame)

    #get the list of profession
    trainingProfessionList = processedTrainingFrame['Profession'].unique()
    
    #create a label binary format
    lbFormat = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

    #Extract the profession column and join the data frame
    processedTrainingFrame = processedTrainingFrame.join(pd.DataFrame(lbFormat.fit_transform(processedTrainingFrame['Profession']),columns=lbFormat.classes_,index=processedTrainingFrame.index))

    #Add a new column -> Other Profession
    dummyData = [0] * len(processedTrainingFrame['Profession'])
    processedTrainingFrame['Other Profession'] = dummyData

    #Remove the profession Column
    processedTrainingFrame = processedTrainingFrame.drop(['Profession'], axis = 1)

    return processedTrainingFrame.drop(['Income in EUR'],axis = 1), processedTrainingFrame['Income in EUR']


def PreprocessingTestDS():
    testFrame = pd.read_csv(testDS_path)

    #Remove income Column
    testFrame = testFrame.drop(['Income'],axis = 1)

    processedTestFrame = Preprocessing(testFrame)

    #Formate the profession column
    trainingFrame = pd.read_csv(trainDS_path)
    refProfessionList = list(set(processedTestFrame['Profession'].unique()) - set(trainingFrame['Profession'].unique()))
    processedTestFrame['Profession'] = processedTestFrame['Profession'].replace(refProfessionList,'Other Profession')

    #create a label binary format
    lbFormat = LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)

    #Extract the profession column and join the data frame
    processedTestFrame = processedTestFrame.join(pd.DataFrame(lbFormat.fit_transform(processedTestFrame['Profession']),columns=lbFormat.classes_,index=processedTestFrame.index))

    #Add a new columns to match the training frame
    dummyData = [0] * len(processedTestFrame['Profession'])
    neededProfessionList = list(set(trainingFrame['Profession'].unique()) - set(processedTestFrame['Profession'].unique()))

    for prof in neededProfessionList:
        processedTestFrame[prof] = dummyData

    #Remove the profession Column
    processedTestFrame = processedTestFrame.drop(['Profession'], axis = 1)
    return processedTestFrame

def ModelCreation(xFrame, yFrame):
    return LinearRegression().fit(xFrame,yFrame)

def run():

    print("started training data preprocessing")
    #load and preprocess training data
    (xDataFrame,yDataFrame) = PreprocessingTrainingDS()
    
    print("started creating linear model")
    #create model and train
    linearModel = ModelCreation(xDataFrame,yDataFrame)

    print("started training data preprocessing")
    #load and preprocess test data
    testDataFrame = PreprocessingTestDS()

    print("started prediction")
    #prediction
    prediction = linearModel.predict(testDataFrame)
    
    print("pusing to a file") 
    numpy.savetxt('predictedOutput.csv',prediction)

    print("Mean Square Error is " + str(linearModel.score(testDataFrame,prediction)))
    

if __name__ == '__main__':
    run()
