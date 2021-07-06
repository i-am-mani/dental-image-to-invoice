from image_segment import ImageSegementation
import tensorflow as tf
import cv2
import pickle
import numpy as np
import base64

image_segementation = ImageSegementation('opencv-models/background.jpeg')
image_segementation.init_hed()

inception_model = tf.keras.models.load_model('models/incep_cnn_v1.h5')

def predict_image(img_path):
    sample = cv2.imread(img_path)
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
    if(sample.shape[0] > 1000): # resize to reduce to speed up edge detection
        sample = cv2.resize(sample, (sample.shape[1]//2,sample.shape[0]//2))
    
    rects = image_segementation.segment(sample)

    with open('models/label_encoder.p', 'rb') as file:
        labelEncoder = pickle.load(file)
    
    predictions = []
    for (x1,y1, w, h) in rects:
        im_test = sample[y1:y1+h, x1:x1+w]
        im_test = cv2.resize(im_test, (224,224))

        model_input = np.reshape(im_test, (-1,224,224,3))
        res = inception_model.predict(model_input)
        res = np.squeeze(res)
        prediction = res.argmax()
        prediction = labelEncoder.inverse_transform([prediction])

        encoded_images = base64.b64encode(cv2.imencode('.jpg', cv2.cvtColor(im_test, cv2.COLOR_BGR2RGB))[1]).decode()

        predictions.append(
            {'image': f'data:image/png;base64, {encoded_images}', 
             'prediction': prediction[0]})
    return predictions