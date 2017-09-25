#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv

import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import numpy as np
from deal import preData

# 数据目录
data_dir = 'data'

w2v_file = 'word2vec/vectors.bin.3g200dim'

with open(data_dir + '/d.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile)
    with open(data_dir + '/data.txt', 'rb') as filein:
        for line in filein:
            line_list = line.strip('\n').split(' ')
            spamwriter.writerow(line_list)

data_set = pd.read_csv(data_dir + '/d.csv')

p = preData.preData()
p.load_word_vectors(w2v_file)
p.count_word_tag(data_set)
p.count_word_subject(data_set)

print "len word_vector_dict = ", len(p.word_vector_dict)
print "len tag_count = ", p.tag_count
print "len sub_count = ", p.sub_count

X, y = p.transform_data(data_set)

print "X,Y", X, y
print '\n'

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print X_train.shape, Y_train.shape, X_test.shape, Y_test.shape

# TRAIN LR MODEL,训练逻辑斯蒂模型
lr = linear_model.LogisticRegression(C=1e5)
lr.fit(X_train, Y_train)

# PREDICT,在测试集上进行预测
Y_pred = lr.predict(X_test)

print "Y_pred = ", Y_pred
print "Y_test = ", Y_test

true_false = (Y_pred == Y_test)
accuracy = np.count_nonzero(true_false)/float(len(Y_test))
# acc_log = round(lr.score(Y_test, Y_pred) * 100, 2)
print "accuracy is %f" % accuracy
