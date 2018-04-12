import re, os, glob, random, csv, time, json, yaml

import pandas as pd
import numpy as np
from openpyxl import load_workbook

from sklearn.preprocessing import OneHotEncoder, Binarizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import  Pipeline

import jieba
import jieba.posseg as pseg
#jieba.load_userdict('./hardskill.dict')


# from keras.layers import Input, Dense, concatenate, Dropout, Activation
# from keras.models import Model, load_model, Sequential

import pickle

# from utility import plot_confusion_matrix, multiclass_metric
#from resume_classifier import ResumeClassifier

import itertools
#from gensim.models import KeyedVectors

import warnings; warnings.simplefilter('ignore')
os.chdir("/Users/zevil/Downloads/projects/magpie-master")


class ResumeClassifier():
    tk = jieba.Tokenizer()

    def __init__(self, model_path=None, text_path=None, date_frame=None,
                 category_change_path=None, stopwords_path=None):
        self.__jds = None
        self.__trainable = False
        self.__category_change = None
        self.__stopwords = None
        self.__model = None
        self.__level_2_tokenizer = None
        self.__level_3_tokenizer = None

        if (model_path is not None) + (text_path is not None) + (date_frame is not None) != 1:
            raise "One and ONLY one of parameter text_path or model_path, df_frame is must"

        if model_path:
            with open(model_path, 'rb') as f:
                self.__model = pickle.load(f)
                self.__trainable = False
        else:
            if text_path:
                jds = self.__load_jd_from_text(text_path)
                jds.replace(['\\n| |\\xa0|\/|\（|\）|\-'], [''], regex=True, inplace=True)
                #                 self.__jds = jds[jds.category1 == '技术']
                self.__jds = jds
            else:
                must_cols = set(['category1', 'category2', 'category3', 'detail', 'job_name'])
                df_cols = set(date_frame.columns.tolist())
                if not must_cols.issubset(df_cols):
                    raise 'the dataframe must contain following columns: ' + str(must_cols)

                self.__jds = date_frame

            self.__trainable = True
            if stopwords_path:
                self.__stopwords = self.__load_stop_words(stopwords_path)
            if category_change_path:
                self.__category_change = self.__load_category_change(category_change_path)

    @classmethod
    def from_pretrained_model(cls, file):
        return cls(model_path=file)

    @classmethod
    def from_text(cls, file, category_change_path=None, stopwords_path=None, ):
        return cls(text_path=file, category_change_path=category_change_path, stopwords_path=stopwords_path)

    @classmethod
    def from_df(cls, df, category_change_path=None, stopwords_path=None):
        return cls(date_frame=df, category_change_path=category_change_path, stopwords_path=stopwords_path)

    @staticmethod
    def default_tokenizer(string):
        tk = ResumeClassifier.tk
        tk.add_word('大数据')
        tk.add_word('.net')
        tk.add_word('机器学习')
        return tk.cut(string)

    def __load_jd_from_text(self, folder):
        abspth = os.path.abspath(folder)
        files = glob.glob(abspth + '/**/*.txt', recursive=True)
        res = list()
        for file in files:
            jd = self.__extract_file_fields(file)
            res.append(jd)
        return pd.DataFrame(res)

    def __extract_file_fields(self, file):
        jd = dict()
        # '技术_人工智能_图像处理_2567485.txt'
        fields = os.path.basename(file).split('_')
        jd['id'] = (fields[-1])[:-4]  # Truancate '.txt'
        jd['category1'] = fields[0]
        jd['category2'] = fields[1]
        jd['category3'] = fields[2] if len(fields) > 3 else None

        regex = r'\n(.*?)\n(.*?)\n( \(该职位已下线\) )?\n+(.*?)\n+(.*?)\n职位诱惑：\n(.*?)\n+职位描述：\n+(.*)\n+工作地址(.*?)查看(完整)?地图'
        pattern = re.compile(regex, re.MULTILINE | re.DOTALL)
        with open(file, 'r+') as f:
            mo = re.search(pattern, f.read())
            if mo is None:
                return jd
            jd['company'] = mo.group(1)
            jd['job_name'] = mo.group(2)
            if mo.group(3):
                jd['is_done'] = True
            else:
                jd['is_done'] = False
            try:
                # '7k-14k /厦门 / 经验1-3年 / 本科及以上 / 全职'
                jd['salary'], jd['location'], jd['experience'], jd['education'], jd['contract_type'] = mo.group(
                    4).split('/')
            except ValueError as e:
                print(f.name)
                raise e
            jd['lables'] = mo.group(5)
            jd['advantage'] = mo.group(6)
            jd['detail'] = mo.group(7)
            jd['address'] = mo.group(8)
        return jd

    def __load_category_change(self, path):
        with open(path, 'r') as f:
            m = yaml.load(f)
        return m

    def __load_stop_words(self, path):
        if not os.path.isfile(path):
            raise ("Fail to find file {}".format(path))
        stopwords = list()
        with open(path, 'r') as f:
            for line in f:
                stopwords.append(line.strip())
        return stopwords

    def __etl(self):
        # low the case first, because in the yaml, all fields is lower case
        for col in ['category1', 'category2', 'category3', 'detail', 'job_name']:
            self.__jds[col] = self.__jds[col].str.lower()

        # Apply the category change
        cate2_change = self.__category_change['category2_change']
        cate3_change = self.__category_change['category3_change']
        self.__jds['cate3_bello'] = self.__jds.category3.apply(
            lambda x: cate3_change[x] if x in cate3_change.keys() else x)
        self.__jds['cate2_bello'] = self.__jds.category2
        for cate3, cate2 in cate2_change.items():
            self.__jds.loc[self.__jds.cate3_bello == cate3, 'cate2_bello'] = cate2

        # Combine the cate1 and cate2, because the model doesn't predict cate1
        self.__jds['cate2_bello'] = self.__jds.category1 + '_' + self.__jds.cate2_bello

        # Drop rows with too few category
        filtered = self.__jds.groupby('cate3_bello').filter(lambda x: len(x) >= 30)
        self.__jds = filtered

    def __split_dataset(self, test_size=0.33):
        """
        Split test and train set even for both svm model and knn model
        :param test_size:
        :return:
        """
        dataset = {}
        X_train, X_test = train_test_split(self.__jds, stratify=self.__jds.cate3_bello, test_size=test_size)

        dataset['svm'] = {'X_train': X_train['job_name'],
                          'X_test': X_test['job_name'],
                          'y_train': X_train['cate2_bello'],
                          'y_test': X_test['cate2_bello']}

        for cate2 in X_train.cate2_bello.unique():
            train = X_train[X_train.cate2_bello == cate2]
            test = X_test[X_test.cate2_bello == cate2]
            dataset[cate2] = {'X_train': train.detail,
                              'X_test': test.detail,
                              'y_train': train.cate3_bello,
                              'y_test': test.cate3_bello}

        self.__dataset = dataset

    def __build_svm_pipe(self):
        if self.__stopwords:
            tfidf = TfidfVectorizer(stop_words=self.__stopwords,
                                    tokenizer=ResumeClassifier.default_tokenizer,
                                    lowercase=False)
        else:
            tfidf = TfidfVectorizer(tokenizer=ResumeClassifier.default_tokenizer,
                                    lowercase=False)

        ovr_clf = OneVsRestClassifier(SVC(probability=True, kernel='linear'))
        svm_pipe = Pipeline(steps=[('tfidf', tfidf), ('svm', ovr_clf)])
        return svm_pipe

    def __build_svm_pipe2(self, k=500):
        cv = CountVectorizer(stop_words=self.__stopwords,
                             tokenizer=ResumeClassifier.default_tokenizer)
        sk = SelectKBest(mutual_info_classif, k=k)
        tfidf = TfidfTransformer()
        # ovr  = OneVsRestClassifier(SVC(probability=True, kernel='rbf'))
        svm = LinearSVC()
        pipe = Pipeline(steps=[('cv', cv), ('tfidf', tfidf), ('sk', sk), ('svm', svm)])
        return pipe

    def __build_knn_pipe(self, neighbors=20):
        if self.__stopwords:
            tfidf = TfidfVectorizer(lowercase=False,
                                    stop_words=self.__stopwords,
                                    tokenizer=ResumeClassifier.default_tokenizer)
        else:
            tfidf = TfidfVectorizer(lowercase=False,
                                    tokenizer=ResumeClassifier.default_tokenizer)

        knn_clf = KNeighborsClassifier(n_neighbors=neighbors, weights='distance')
        knn_pipe = Pipeline(steps=[('tfidf', tfidf), ('knn', knn_clf)])
        return knn_pipe

    def __build_model(self):
        model = dict()
        model['svm'] = self.__build_svm_pipe2(k=700)
        for cate2 in self.__dataset.keys():
            if cate2 == 'svm' or cate2[:2] != '技术':
                continue
            model[cate2] = self.__build_svm_pipe2(k=700)

        self.__model = model

    def prepare_for_fit(self):
        self.__etl()
        self.__split_dataset()
        self.__build_model()

    def fit(self):
        for name, model in self.__model.items():
            X_train = self.__dataset[name]['X_train']
            y_train = self.__dataset[name]['y_train']
            print('---start to train {}'.format(name))
            print(X_train.sample(1))
            print(y_train.sample(1))
            model.fit(X_train, y_train)

    def __etl_for_predict(self, title, detail):
        return title.lower(), detail.lower()

    def predict(self, title, detail):
        title, detail = self.__etl_for_predict(title, detail)
        svm = self.__model['svm']
        cate2 = svm.predict(pd.Series(title))
        cate2 = cate2.tolist()[0]
        knn = self.__model[cate2]
        cate3 = knn.predict(pd.Series(detail))
        cate3 = cate3.tolist()[0]

        return cate2, cate3

    def predict_proba(self, title, detail, level2_m=2, level3_n=2, top_k=4):
        """
        Important: The top_k result is not guaranteed. It's limited by actual category.

        :param title:
        :param detail:
        :param level2_m: select top m probability from level_2 prediction
        :param level3_n: select top n probability from level_3 prediction for each level_2 prediction
        :param top_k:  select top k probability from m * n result
        :return: a  list of top k result, as following:
            [ {major: 技术, minor: 后端, detail:java, score:0.6},
            {major:技术, minor:后端, detail:php, score: 0.3}]
        """

        if top_k > level3_n * level2_m:
            raise "top_k parameter is too big to fullfill"

        title, detail = self.__etl_for_predict(title, detail)
        svm = self.__model['svm']
        cate2 = svm.predict_proba(pd.Series(title))
        cate2 = pd.DataFrame(cate2.T)
        cate2 = cate2.nlargest(level2_m, columns=0).to_dict()[0]

        res = list()
        for c2, c2_proba in cate2.items():
            c2_string = svm.classes_[c2]
            knn = self.__model[c2_string]
            cate3 = knn.predict_proba(pd.Series(detail))
            cate3 = pd.DataFrame(cate3.T)
            cate3 = cate3.nlargest(level3_n, columns=0).to_dict()[0]
            for c3, c3_proba in cate3.items():
                c3_string = knn.classes_[c3]
                tmp = dict()
                tmp['major'] = c2_string.split('_')[0]
                tmp['minor'] = c2_string.split('_')[1]
                tmp['detail'] = c3_string
                tmp['minor_score'] = c2_proba
                tmp['detail_score'] = c3_proba
                tmp['score'] = c2_proba * c3_proba
                res.append(tmp)

        return sorted(res, key=lambda x: x['score'], reverse=True)[0:top_k]

    def score(self):
        score = dict()
        for name, model in self.__model.items():
            dataset = self.__dataset[name]
            score[name] = {'train_accuracy': model.score(dataset['X_train'], dataset['y_train']),
                           'test_accuracy': model.score(dataset['X_test'], dataset['y_test'])}
        return score

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.__model, f)

    def save_category_info(self, path):
        res = list()
        svm = self.__model['svm']
        cate2s = svm.classes_.tolist()
        for cate2 in cate2s:
            knn = self.__model[cate2]
            cate3 = knn.classes_.tolist()
            tmp = dict()
            tmp['name'] = cate2
            tmp['_items'] = cate3
            res.append(tmp)

        with open(path, 'w') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)

    def get_detaset(self):
        return self.__dataset

    def get_model(self):
        return self.__model


excelPath = 'data/term_weight.xlsx'
excelWriter = pd.ExcelWriter(excelPath, engine='openpyxl')
model = pickle.load(open('data/model.pkl', 'rb'))
for name, md in model.items():

    k = 700
    if name == 'svm':
        k = 2000
    print(name)
    words = md.named_steps['cv'].get_feature_names()
    scores = md.named_steps['sk'].scores_
    i = 0
#     print(words)
    weight = {}
    for i, word in enumerate(words):
        weight[word] = scores[i]
#     print(weight)
    select_word = sorted(weight.items(), key=lambda item:item[1], reverse=True)[:k]
    df = pd.DataFrame(select_word, columns=['TERM', 'WEIGHT'])
    df.to_excel(excel_writer=excelWriter, sheet_name=name,index=True)
excelWriter.close()
excelWriter.save()