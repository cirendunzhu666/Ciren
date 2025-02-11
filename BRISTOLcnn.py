import numpy as np
from numpy import mean
from numpy import std
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import MaxPooling1D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import string
import re
import random
import hashlib
import os
from sklearn.metrics import f1_score, precision_recall_curve
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
dataset_path = 'C:/Users/93678/PycharmProjects/Sapienza_data/progetto_bristol/dataset'
model_path = 'C:/Users/93678/PycharmProjects/Sapienza_data/progetto_bristol/model'
if not os.path.exists(model_path):
    os.makedirs(model_path)
save_model = 1

def read_and_norm(dataset_path, type_,type):
    with open('{}/{}/{}.txt'.format(dataset_path, type_,type), 'rb') as f:
        matrix = [[float(x) for x in line.split()] for line in f]
    matrix = np.array(matrix)
    min_m = matrix.min().min()
    max_m = matrix.max().max()
    matrix = ((matrix - min_m) / (max_m - min_m))
    return matrix

def load_full_dataset(type_):
	classification = np.loadtxt('{}/{}/classification.txt'.format(dataset_path, type_))
	classification = np.array(classification).reshape(-1,1)

	with open('{}/{}/hr.txt'.format(dataset_path, type_), "r") as file:
		hr = []
		righe_con_9_colonne = []
		for indice, riga in enumerate(file):
			colonne = riga.split()
			if len(colonne) == 9:
				righe_con_9_colonne.append(indice)
			else:
				hr.append(colonne)
	hr = [[float(string) for string in inner] for inner in hr]
	hr = np.array(hr)


	shape = read_and_norm(dataset_path, type_,'shape')
	el = read_and_norm(dataset_path, type_,'el')
	dist = read_and_norm(dataset_path, type_,'dist')

	classification = np.delete(classification, righe_con_9_colonne, 0)
	shape = np.delete(shape, righe_con_9_colonne, 0)
	el = np.delete(el, righe_con_9_colonne, 0)
	dist = np.delete(dist, righe_con_9_colonne, 0)


	data_X = np.array([p for p in zip(shape, dist, el, hr)])
	data_X = data_X.reshape(data_X.shape[0], data_X.shape[1], data_X.shape[2], 1)

	return(data_X,classification)



def evaluate_model_2dconv(trainX, trainy, testX, testy):
    set_seed(42)  # 固定种子
    verbose, epochs, batch_size = 1, 300, 1
    n_outputs = trainy.shape[1]
    model = Sequential()

    # 模型结构
    model.add(Conv2D(filters=9, kernel_size=(4, 1), input_shape=trainX.shape[1:], activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=3, kernel_size=(1, 3), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_outputs, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=50, restore_best_weights=True)

    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2, callbacks=[es])

    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=1)

    # 预测概率
    pred_label_ = model.predict(testX, batch_size=batch_size).reshape(-1)
    true_labels = testy.reshape(-1)

    # 计算 F1-score 和最佳阈值
    precision, recall, thresholds = precision_recall_curve(true_labels, pred_label_)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = f1_scores[:-1]
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best Threshold: {best_threshold:.3f}, Best F1-score: {best_f1:.3f}")
    print(trainX.shape)

    # 使用最佳阈值进行二分类
    final_predictions = (pred_label_ >= best_threshold).astype(int)

    # 计算基于氨基酸对类别的预测准确率
    residue_name = add_pairs_label('test')
    residue_pairs = pd.DataFrame(residue_name, columns=['resA', 'resB', 'TypeLabel'])
    residue_pairs['TrueLabel'] = true_labels
    residue_pairs['PredLabel'] = final_predictions
    pairs_accuracy_result = residue_pairs.groupby('TypeLabel').apply(lambda group: (group['TrueLabel'] == group['PredLabel']).mean())



    return accuracy, history, best_threshold, best_f1, pairs_accuracy_result
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)



def add_pairs_label(type_):
    with open('{}/{}/{}.txt'.format(dataset_path,type_,'dataset'),'r') as f:
        next(f)  # //跳过第一行
        residue_name = []#储存标签
        for line in f:
            columns = line.strip().split(',')# 按逗号分割每行数据
            selected_name = [columns[1], columns[2]]#取第二列和第三列
            residue_name.append(selected_name)

    # 分类规则
    H = ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TYR', 'TRP']
    P = ['SER', 'PRO', 'THR', 'CYS', 'ASN', 'GLN']
    C = ['HIS', 'LYS', 'ARG', 'ASP', 'GLU']

    for i in range(len(residue_name)):
        if (residue_name[i][0] in H) and (residue_name[i][1] in H):
            label = 'HH'
        elif (residue_name[i][0] in P) and (residue_name[i][1] in P):
            label = 'PP'
        elif (residue_name[i][0] in C) and (residue_name[i][1] in C):
            label = 'CC'
        elif (residue_name[i][0] in H and residue_name[i][1] in P) or (residue_name[i][0] in P and residue_name[i][1] in H):
            label = 'PH'
        elif (residue_name[i][0] in P and residue_name[i][1] in C) or (residue_name[i][0] in C and residue_name[i][1] in P):
            label = 'PC'
        elif (residue_name[i][0] in H and residue_name[i][1] in C) or (residue_name[i][0] in C and residue_name[i][1] in H):
            label = 'HC'
        else:
            label = 'NA'
        residue_name[i].append(label)

    return residue_name

def run_experiment(repeats=1):
    set_seed(42)
    trainX, trainy = load_full_dataset('train')
    testX, testy = load_full_dataset('test')

    scores = []
    best_thresholds = []
    best_f1_scores = []
    pairs_accuracy_results = []

    for r in range(repeats):
        set_seed(42 + r)
        score, history, best_threshold, best_f1, pairs_accuracy_result = evaluate_model_2dconv(trainX, trainy, testX, testy)
        score = score * 100.0
        print(f'>#{r+1}: Accuracy: {score:.3f}, Best F1: {best_f1:.3f}, Best Threshold: {best_threshold:.3f}')
        print("Pairs Accuracy Per Type:")
        print(pairs_accuracy_result)

        scores.append(score)
        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)
        pairs_accuracy_results.append(pairs_accuracy_result)

    return scores, best_thresholds, best_f1_scores, pairs_accuracy_results


run_experiment()