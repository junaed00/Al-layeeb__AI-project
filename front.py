import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import cv2

from matplotlib import pyplot as plt;

st.markdown('<h1 style="color:black;">Hand Gesture Recognition Model</h1>',unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into 20 categories of hand signal</h2>', unsafe_allow_html=True)


def preprocess_input(simg):
    img = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR);
    img = cv2.resize(img, (302, 302))
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, img.shape[0] - 1, img.shape[1] - 1)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    imggc = img * mask2[:, :, np.newaxis]
    mask[0, :], mask[:, 0], mask[:, mask.shape[1] - 1] = (cv2.GC_BGD, ) * 3
    mask3, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    mask4 = np.where((mask3 == 2) | (mask3 == 0), 0, 1).astype('uint8')
    imgf = img * mask4[:, :, np.newaxis]
    tmask = np.stack((mask4[1:-1, 1:-1] * 255, ) * 3, axis=-1)
    # tmask = cv2.flip(tmask, 1)
    tmask = cv2.resize(tmask, (50, 50))
    tmask = tf.keras.applications.vgg16.preprocess_input(tmask)
    return tmask


laeeb = tf.keras.models.load_model("./laeeb/")

upload = st.file_uploader('Insert image for classification', type=['png', 'jpg'])
c1, c2 = st.columns(2)
if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    img = preprocess_input(img)

    c1.header('Input Image')
    c1.image(img, clamp=True)
    c1.write(img.shape)

    img = np.expand_dims(img, 0)
    vgg_preds = laeeb.predict(img)
    vgg_pred_classes = np.argmax(vgg_preds, axis=1)
    c2.header('Output')
    c2.subheader('Predicted class :')
    c2.write(vgg_pred_classes)
