import cv2
import tensorflow as tf
import numpy as np
import json
import time
json_file = "dataKeys.json"
mainModel = tf.keras.models.load_model('firstTry.h5')
subcatModels = {
    "capacitor": tf.keras.models.load_model('modelForCategory_capacitor.h5'),
    "diodo": tf.keras.models.load_model('modelForCategory_diodo.h5'),
    "microcontrolador": tf.keras.models.load_model('modelForCategory_microcontrolador.h5'),
    "rele" : tf.keras.models.load_model('modelForCategory_rele.h5'),
    "resistencia": tf.keras.models.load_model('modelForCategory_resistencia.h5'),
    "sensor": tf.keras.models.load_model('modelForCategory_sensor.h5'),
    "transistor": tf.keras.models.load_model('modelForCategory_transistor.h5')
}

 
with open(json_file, "r") as json_file:
    data_dict = json.load(json_file)

def preprocess_frame(frame):
    input_size = (100,100)
    resized_frame = cv2.resize(frame, input_size)
    normalized_frame = resized_frame / 255.0
    input_frame = tf.expand_dims(normalized_frame, axis=0)
    return input_frame

cap  = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se puede abrir la camara")
    exit()
while True:
    ret, frame = cap.read()
    if not ret:
        print("no se pudo leer el frmae")
        break
    x, y, w, h = 100, 100, 200, 200
    roi = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    input_frame = preprocess_frame(roi) #frame
    
    predictions = mainModel.predict(input_frame)
    predicted_class = np.argmax(predictions, axis=1)
    prediction_name = data_dict[str(predicted_class[0])]['name']
    predictionsText = f"Prediction: { prediction_name }"
 
    # try:
      
    #     predictSubClass = subcatModels[prediction_name].predict(input_frame)

    #     predicted_subCat = np.argmax(predictSubClass, axis = 1)
    #     predictionSubClassName = data_dict[str(predicted_class[0])]['types'][str(predicted_subCat[0])]
    #     print(predictionSubClassName)
    #     predictionsText = predictionsText + " \n" + predictionSubClassName
       
    # except:
    #     pass
    cv2.putText(frame, predictionsText, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Trabajo final', frame)
    time.sleep(0.5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

