import face_recognition
from face_recognition import face_locations
import os
import sys
from tqdm import tqdm
import scipy.io
import pickle
import numpy as np
import pandas as pd 

sys.path.append('DB + Model/gi.py')
#from gi import *

# Set data paths
data_dir = '/Users/arthur/Desktop/2022 EM/Machine Learning/Projet/DB + Model'
model_dir = os.path.join(data_dir,'model.pkl')
img_dir = os.path.join(data_dir,'df_2')
indices_path = os.path.join(data_dir, 'indices_train_test.mat')
attr_path = os.path.join(data_dir, 'lfw_att_73.mat')

# Load model
loaded_model = pickle.load(open(model_dir, 'rb'))

# Load attributes
attr_name_mat = '/Users/arthur/Desktop/2022 EM/Machine Learning/Projet/DB + Model/LWFA+/attrname.mat'
attr_name = scipy.io.loadmat(attr_name_mat)['AttrName']
attr_name = [str(s[0]) for s in attr_name.tolist()[0]]

# Photo names 
FichList = [ f for f in os.listdir('/Users/arthur/Desktop/2022 EM/Machine Learning/Projet/DB + Model/df_2') if os.path.isfile(os.path.join('/Users/arthur/Desktop/2022 EM/Machine Learning/Projet/DB + Model/df_2',f)) ]

# Create Df 
df_label = pd.DataFrame(columns=attr_name, index=FichList)


# Functions 
def face_recog():
    global vecs
    global fnames
    global predict_score
    global df_2
    global df_label
    vecs = []
    fnames = []
    i = 0
    for fname in tqdm(df_label.index):
        i += 1
        img_path = os.path.join(img_dir, fname)
        X_img = face_recognition.load_image_file(img_path)
        X_faces_loc = face_recognition.face_locations(X_img)
        if len(X_faces_loc) != 1:
            continue
        faces_encoding = face_recognition.face_encodings(X_img, known_face_locations=X_faces_loc)[0]
        
        vecs.append(faces_encoding)
        fnames.append(fname)
        
    df_feat = pd.DataFrame(vecs, index=fnames)
    df_label = df_label[df_label.index.isin(df_feat.index)]
    df_feat.sort_index(inplace=True)
    df_label.sort_index(inplace=True)
    predict_score = loaded_model.predict_proba(vecs)
    df_2 = pd.DataFrame(predict_score, columns=df_label.columns, index=fnames)

def df2_sort() :
    global list_trait
    df_2['Female'] = 1 - df_2['Male']
    df_score = df_2[['Male','Female', 'Asian', 'Black', 'White','Indian','Black Hair','Blond Hair','Brown Hair','Curly Hair','Wavy Hair','Straight Hair','Oval Face','Square Face','Round Face','Baby', 'Child', 'Youth', 'Middle Aged', 'Senior']]
    df_score['Gender'] = df_score[['Male', 'Female']].idxmax(axis=1)
    df_score['Race'] = df_score[['Asian', 'Black', 'White','Indian']].idxmax(axis=1)
    df_score['Hair color'] = df_score[['Black Hair','Blond Hair','Brown Hair']].idxmax(axis=1)
    df_score['Hair Shape'] = df_score[['Curly Hair','Wavy Hair','Straight Hair']].idxmax(axis=1)
    df_score['Face Shape'] = df_score[['Oval Face','Square Face','Round Face']].idxmax(axis=1)
    df_score['Looking Age'] = df_score[['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior']].idxmax(axis=1)

    df_trait = df_score[['Gender', 'Race', 'Hair color', 'Hair Shape', 'Face Shape', 'Looking Age']]
    list_trait = df_trait[['Gender', 'Race', 'Hair color', 'Hair Shape', 'Face Shape', 'Looking Age']].value_counts()[:1].index.tolist()
    print(list_trait)


face_recog()
df2_sort()



#input = ['Female', 'Black', 'Blond Hair', 'Wavy Hair', 'Oval Face', 'Child']
