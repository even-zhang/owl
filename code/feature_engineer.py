import pickle, os, traceback
import jieba
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2

os.chdir("/Users/zevil/Downloads/projects/owl")
data = pickle.load(open('data/data.pkl', 'rb'))
st_path = "/Users/zevil/Downloads/projects/owl/data/hanzi.txt"


def load_stop_words(path):
    if not os.path.isfile(path):
        raise ("Fail to find file {}".format(path))
    stopwords = list()
    with open(path, 'r') as f:
        for line in f:
            stopwords.append(line.strip())
    return stopwords


stopwords = load_stop_words(st_path)


def cut_word(string):
    words = jieba.cut(string)
    output = ''
    for word in words:
        if word not in stopwords:
            output += word
            output += ' '
    return [output]


excelPath = 'data/term_weight.xlsx'
excelWriter = pd.ExcelWriter(excelPath, engine='openpyxl')

for name, item in data.items():
    # corpus = jieba.cut(item.X_train)
    k = 700
    if name == 'svm':
        k = 2000
    if name != 'svm' and name[:2] != '技术':
        continue
    cv = CountVectorizer(tokenizer=jieba.cut, stop_words=stopwords)
    tfidf = TfidfTransformer()
    X = cv.fit_transform(list(item['X_train']))
    # corpus = list(map(cut_word, list(item['X_train'])))
    Y = list(item['y_train'])
    X_tfidf = tfidf.fit_transform(X)
    # sk = SelectKBest(chi2, k=k)
    sk = SelectKBest(chi2, k=k).fit(X, Y)

    words = cv.get_feature_names()
    new_X = sk.transform(X)
    scores = sk.scores_
    i = 0
    #     print(words)
    weight = {}
    for i, word in enumerate(words):
        weight[word] = scores[i]
    #     print(weight)
    select_word = sorted(weight.items(), key=lambda d: d[1], reverse=True)[:k]
    df = pd.DataFrame(select_word, columns=['TERM', 'WEIGHT'])
    df.replace(['\b'], [''], regex=True, inplace=True)
    try:
        df.to_excel(excel_writer=excelWriter, sheet_name=name, index=True)
    except:
        # df = df.applymap(lambda x: x.encode('unicode_escape').
        #                  decode('utf-8') if isinstance(x, str) else x)
        # df.to_excel(excel_writer=excelWriter, sheet_name=name, index=True)
        print(traceback.format_exc())
        continue
#
excelWriter.close()
excelWriter.save()