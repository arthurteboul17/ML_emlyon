from tkinter import PhotoImage
import streamlit as st 
import sys
import numpy as np
import pandas as pd
import pickle


sys.path.append('DB + Model/main.py')

from main import *

st.title("Project")

loaded_model = pickle.load(open(model_dir, 'rb'))

input_photo = st.file_uploader('Upload a photo')


if input_photo is not None : 
    st.image(input_photo)
    file_path = '/Users/arthur/Desktop/2022 EM/Machine Learning/Projet/DB + Model/input/'
    photo = input_photo.name
    fich_name = [photo]
    df_label_input = pd.DataFrame(columns=attr_name, index=fich_name)
    
    st.write("Filename: ", photo)
    
    vecs = []
    fnames = []
    i = 0
    for fname in tqdm(df_label_input.index):
        i += 1
        img_path = input_photo
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
    predict_score_input = loaded_model.predict_proba(vecs)
    df_input = pd.DataFrame(predict_score_input, columns=df_label.columns, index=fnames)
    #st.write(df_input)

    df_input['Female'] = 1 - df_2['Male']
    df_score_input = df_input[['Male','Female', 'Asian', 'Black', 'White','Indian','Black Hair','Blond Hair','Brown Hair','Curly Hair','Wavy Hair','Straight Hair','Oval Face','Square Face','Round Face','Baby', 'Child', 'Youth', 'Middle Aged', 'Senior']]
    df_score_input['Gender'] = df_score_input[['Male', 'Female']].idxmax(axis=1)
    df_score_input['Race'] = df_score_input[['Asian', 'Black', 'White','Indian']].idxmax(axis=1)
    df_score_input['Hair color'] = df_score_input[['Black Hair','Blond Hair','Brown Hair']].idxmax(axis=1)
    df_score_input['Hair Shape'] = df_score_input[['Curly Hair','Wavy Hair','Straight Hair']].idxmax(axis=1)
    df_score_input['Face Shape'] = df_score_input[['Oval Face','Square Face','Round Face']].idxmax(axis=1)
    df_score_input['Looking Age'] = df_score_input[['Baby', 'Child', 'Youth', 'Middle Aged', 'Senior']].idxmax(axis=1)

    df_trait_input = df_score_input[['Gender', 'Race', 'Hair color', 'Hair Shape', 'Face Shape', 'Looking Age']]
    list_trait_input = df_trait_input[['Gender', 'Race', 'Hair color', 'Hair Shape', 'Face Shape', 'Looking Age']].value_counts()[:1].index.tolist()
    st.write('list_trait', list_trait[0])
    st.write('list_trait_input',list_trait_input[0])

    common_t = np.intersect1d(list_trait, list_trait_input) 
    if list_trait_input[0][0] == 'Male' : 
        st.title('Output: Dislike')
    elif len(common_t) >= 4:
        st.title('Output: Like')
    else: 
        st.title('Output: rDislike')
