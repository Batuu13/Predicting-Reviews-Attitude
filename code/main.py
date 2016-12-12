import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import glob
import re

def getTestList():
    negativeList = glob.glob("C:/Users/ACER-PC\PycharmProjects\homework2/MRDataset/test/neg/*.txt")
    positiveList = glob.glob("C:/Users/ACER-PC\PycharmProjects\homework2/MRDataset/test/pos/*.txt")
    testData = []
    for item in negativeList[:test_size]:
        f = open(item, encoding="utf8", mode='r')
        testData.append(str(f.read()))
        f.close()
    for item in positiveList[:test_size]:
        f = open(item, encoding="utf8", mode='r')
        testData.append(str(f.read()))
        f.close()
    del negativeList  # Saving memory
    del positiveList  # Saving memory
    return testData

vectorizer = CountVectorizer(min_df=0.01,ngram_range=(1,2))
countVectorizer = CountVectorizer(min_df=0.01,ngram_range=(1,2))
transformer = TfidfTransformer()
######## GETING DATA #######
print("Initialized: Data is getting read.")
negativeList = glob.glob("C:/Users/ACER-PC\PycharmProjects\homework2/MRDataset/train/neg/*.txt")
positiveList = glob.glob("C:/Users/ACER-PC\PycharmProjects\homework2/MRDataset/train/pos/*.txt")
trainData1 = []
trainData2 = []
size = 12500
test_size = 12500

for item in negativeList[:size]:
    f = open(item, encoding="utf8", mode='r')
    trainData1.append(str(f.read()))
    f.close()
for item in positiveList[:size]:
    f = open(item, encoding="utf8", mode='r')
    trainData2.append(str(f.read()))
    f.close()

regex = r"C:/Users/ACER-PC\\PycharmProjects\\homework2/MRDataset/train/\w+\\[0-9]+_([0-9]+)\.txt"
finder = re.compile(regex)

for i in range(len(negativeList)):
    negativeList[i] = int(finder.match(negativeList[i]).group(1))
for i in range(len(positiveList)):
    positiveList[i] = int(finder.match(positiveList[i]).group(1))
#del negativeList # Saving memory
#del positiveList # Saving memory
print("Initialized: Data is read.")
##########################
####### BAG OF WORDS & ###
print("Bag Of Words is started.")
probList = vectorizer.fit_transform(trainData1 + trainData2).toarray()
countVectorizer.fit(trainData1) #negative
del trainData1
unique_names = vectorizer.get_feature_names()    # count of the unique words
counts_neg_dic = countVectorizer.vocabulary_  # count of the total words in spesific class
countVectorizer.fit(trainData2) #positive
del trainData2
counts_pos_dic = countVectorizer.vocabulary_

print("Bag Of Words is finished.")
#########################
######## POSSIBILITES ###
print("Calculating Possibility table")
possibilityListNeg = np.matrix(probList[:size])
possibilityListNeg = transformer.fit_transform(probList[:size]).toarray()

# Change the 2k value on the formula to differ the effect. Or remove the addition to remove the effect completely.

for i in range(len(possibilityListNeg)):
    possibilityListNeg[i] = possibilityListNeg[i] + 1 / ((positiveList[i]) * 2000) # Ratings Affect

possibilityListNeg = possibilityListNeg.sum(axis=0)
possibilityListPos = np.matrix(probList[size:])
possibilityListPos = transformer.fit_transform(probList[size:]).toarray()

for i in range(len(possibilityListPos)):
    possibilityListPos[i] = possibilityListPos[i] + 1 / ((11 - positiveList[i]) * 2000) # Ratings Affect

possibilityListPos = possibilityListPos.sum(axis=0)
del probList

unique_size = len(unique_names)
for i in range(0, len(possibilityListNeg)):
    possibilityListNeg[i] = (possibilityListNeg[i] + 1) / (unique_size + counts_neg_dic.get(unique_names[i],0))
for i in range(0, len(possibilityListPos)):
    possibilityListPos[i] = (possibilityListPos[i] + 1) / (unique_size + counts_pos_dic.get(unique_names[i],0))
print("Possibility table is ready.")
##########################
########## NAIVE BAYES ###
print("Naive Bayes is started.")
#sample = ["I'd read some gushing reviews about this film in the press that claimed it was a masterpiece of originality, so I decided to go watch it. My gut instinct with superhero films is that if they're made by Marvel they are very formulaic and if they're made by DC they tend to be an overlong mess. This film just confirmed that I should trust my gut instinct. Benedict Cumberbatch is well cast, in fact I don't think I could quibble with any of the actors performances, but the script was a typical superhero origin film with no surprises. They gave Dr Strange spiritual powers instead of the usual medical experiment gone wrong scenario, but that just added to the feeling of overall silliness about this film. Nothing in the script made me care about any of the characters and plot-wise this was an attempt at ripping off the Matrix but with magic instead of Kung fu etc. And for the umpteenth time in one of these films it ended with a big CGI fight to save the world. I mean really? Can't these people come up with anything different? The sight of Cumberbatch and the rest running around the city streets in their silly capes and LOTR style costumes was laughable. I've seen some terrible films this year, like Suicide Squad, Star Trek Beyond, Ghostbusters, Batman v Superman, Independence Day, Magnificent Seven, to name just a few. What a complete lack of originality we're having dumped on us this year. I think Independence Day was the worst one but this runs it pretty close. Really naff."]
sample = getTestList()
values = vectorizer.transform(sample).toarray()
sample_length = len(sample)
del sample
correct = 0
wrong = 0
for k in range(0,sample_length):

    neg_possibility = 0
    pos_possibility = 0
    neg_possibility =  np.dot(values[k],np.log(possibilityListNeg))
    pos_possibility = np.dot(values[k],np.log(possibilityListPos))

    sys.stdout.write("\rProgress: %.2f%%" % (k * 100 / sample_length))

    if neg_possibility < pos_possibility and k > size: #POSITIVE
        correct += 1
       # print("Positive #" ,k)
    elif neg_possibility >= pos_possibility and k <= size: #NEGATIVE
       # print("Negative #", k)
        correct += 1


wrong = test_size * 2 - correct
print("\nNaive Bayes is finished.")
print("Correct: ", correct)
print("Wrong: ", wrong)
print("Accuracy", (100*correct)/(test_size * 2))
##########################