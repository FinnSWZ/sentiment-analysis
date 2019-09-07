import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.model_selection import train_test_split
import codecs

#Load train data


def read_file(path):
    array = []
    sum_tap = []
    word = []
    file = open(path)
    contents = file.readlines()
    for content in contents:
        content = content.strip('\n')
        temp = content.split(" ")
        array.append(temp)    
    #array = array[:10000]                         # Choose the number of train sentences
    
    for seq in range(len(array)):
        tap = []
        if array[seq] == ['']:
            pass
        else:
            for seq_2 in range(len(array[seq])):
                x = array[seq][seq_2].split('|')
                word.append(x[0])
                tap.append(x[2])
            sum_tap.append(tap)   
    return array,word,sum_tap

# load test file
def load_test_file(path):
    f = codecs.open(path, mode='r', encoding='utf-8')  
    line = f.readline()   
    word_list = []
    tap_list = []
    word_test = []
    sentence_number = []
    while line:
        a = line.split()
        b = a[0:1]   
        c = a[1:2]
        word_list.append(b)  
        tap_list.append(c)
        line = f.readline()
    f.close()

    tag_test = []
    sentence_tag = []
    word_sentence = []
    for m in word_list:
        if m !=[]:
            word_test += m
            word_sentence += m
        else:
            sentence_number.append(word_sentence)
            word_sentence = []
    for n in tap_list:
        if n != []:
            tag_test += n
        else:
            sentence_tag.append(tag_test)
            tag_test = []
    return word_test,sentence_number,sentence_tag


# feature extract
def feature_extract(array,word):
    sequence = 0
    total_feature = []
    for i in range(len(array)):
        sentence_feature = []
        if array[i] == ['']:
            pass
        else:      
            for j in range(len(array[i])):
                if j == len(array[i])-1:
                    feature = dict(previous_word2=word[sequence-2], previous_word=word[sequence-1], current_word=word[sequence])
              
                elif j == 0:
                    feature = dict(next_word2=word[sequence+2],next_word=word[sequence+1],current_word=word[sequence])  
                else:
                    feature = dict(previous_word=word[sequence-1], current_word=word[sequence], next_word=word[sequence+1])
              
                sentence_feature.append(feature)
                sequence = sequence + 1
            total_feature.append(sentence_feature)
    return total_feature




#Train model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
array,word,y_train = read_file('aij-wikiner-en-wp2')    
x_train = feature_extract(array,word)
crf.fit(x_train,y_train)

# Predict the result
word_test, sentence_number,y_test = load_test_file('wikigold.conll.txt')
x_test = feature_extract(sentence_number,word_test)
y_pred = crf.predict(x_test)
p =  metrics.flat_f1_score(y_test,y_pred,average = 'weighted')
print('Training sentence:' + str(len(array)))
print('Accurate:'+ str(p))

f=open('final_predict.txt','a',newline='')
for a in range(len(y_pred)):
    for b in range(len(y_pred[a])):
        f.write(y_pred[a][b] + '\n')
    f.write('\n')
f.close()