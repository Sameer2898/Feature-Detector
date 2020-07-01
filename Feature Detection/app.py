from cv2 import cv2
from PIL import Image, ImageEnhance
import numpy as np
import streamlit as st
import os
import emoji

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(r'haarcascade_smile.xml')

#Detect faces
def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img, faces

#Detect eyes
def detect_eyes(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return img

#Detect smiles
def detect_smiles(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    smiles = smile_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in smiles:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img

#Cartoonize image
def cartonize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

#Canny Image
def cannize_image(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

#Main code of the face detection
def main():
    st.title('Feature Detector')
    st.text('Create with STREAMLIT and OPEN-CV')
    activities = ['Detection', 'About', 'Contact-Us']
    choice = st.sidebar.selectbox('Select Activity', activities)
    if choice == 'Detection':
        st.subheader('Feature Detector')
        image_file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            # st.write(type(our_image))
            st.image(our_image)
        
        enhance_type = st.sidebar.radio('Enhance Type', ['Original', 'Gray-Scale', 'Contrast', 'Brightness', 'Blurring'])
        if enhance_type == 'Gray-Scale':
            new_img = np.array(our_image.convert('RGB'))
            img = cv2.cvtColor(new_img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # st.write(new_img)
            st.image(gray)
        
        elif enhance_type == 'Contrast':
            c_rate = st.sidebar.slider('Contrast', 0.5, 3.5)
            enhancer = ImageEnhance.Contrast(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)

        elif enhance_type == 'Brightness':
            c_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
            enhancer = ImageEnhance.Brightness(our_image)
            img_output = enhancer.enhance(c_rate)
            st.image(img_output)
        
        elif enhance_type ==  'Blurring':
            new_img = np.array(our_image.convert('RGB'))
            blur_rate = st.sidebar.slider('Brightness', 0.5, 3.5)
            img = cv2.cvtColor(new_img, 1)
            blur_img = cv2.GaussianBlur(img, (11, 11), blur_rate)
            st.image(blur_img)
        
        elif enhance_type == 'Original':
            st.image(our_image, width=300)
        else:
            st.image(our_image, width=300)
        
        #Face Detection
        task = ['Faces', 'Smiles', 'Eyes', 'Cannize', 'Cartonize']
        feature_choice = st.sidebar.selectbox('Find Features', task)

        if st.button('Process'):
            if feature_choice == 'Faces':
                result_img, result_faces = detect_faces(our_image)
                st.image(result_img)
                st.success(f'Found {len(result_faces)} faces.') 
            
            elif feature_choice == 'Smiles':
                result_img = detect_smiles(our_image)
                st.image(result_img)
            
            elif feature_choice == 'Eyes':
                result_img = detect_eyes(our_image)
                st.image(result_img)
            
            elif feature_choice == 'Cartonize':
                result_img = cartonize_image(our_image)
                st.image(result_img)
            
            elif feature_choice == 'Cannize':
                result_canny = cannize_image(our_image)
                st.image(result_canny)

    elif choice == 'About':
        st.subheader('About Face Detection')
        st.markdown('Built with the use of STREAMLIT and OPEN-CV by "SAMEER GOEL".') 
        st.text('Sameer Goel')
        st.success('Always Be Happy')
        st.markdown(emoji.emojize(':grinning_face_with_big_eyes::grinning_face_with_big_eyes:'))
    
    elif choice == 'Contact-Us':
        st.header('Contact Me')
        st.text("Don't try to contact")
        st.text(emoji.emojize(':winking_face_with_tongue:'))

                 
if __name__ == "__main__":
    main()