import os
from keras.layers import Dense
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import tensorflow as tf



def train():
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    data_path = "../prepdat"
    bath_size = 64
    num_epoch = 70
    lr = 0.0001
    num_classes = 8
    nb_test_samples = 3573
    nb_val_samples = 3588
    nb_train_samples = 28558
    save_model_dir_path = 'model_weights/model_sw_3_weights.hdf5'
    train_data_generator = ImageDataGenerator(rescale = 1./255, shear_range=0.2,
                                              zoom_range=0.2, rotation_range=30,
                                              horizontal_flip=True)
    val_data_generator = ImageDataGenerator(rescale=1./255)
    test_data_generator = ImageDataGenerator(rescale=1./255)

    train_generator = train_data_generator.flow_from_directory(os.path.join(data_path, 'train'),
                                                               target_size=(48,48),
                                                               color_mode='grayscale',
                                                               batch_size=bath_size,
                                                               class_mode='categorical')

    val_generator = val_data_generator.flow_from_directory(os.path.join(data_path, 'val'),
                                                           target_size=(48,48),
                                                           color_mode='grayscale',
                                                           batch_size=bath_size,
                                                           class_mode='categorical')

    test_generator = test_data_generator.flow_from_directory(os.path.join(data_path, 'test'),
                                                             target_size=(48, 48),
                                                             color_mode='grayscale',
                                                             batch_size=bath_size,
                                                             class_mode='categorical')
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1), kernel_regularizer=l2(0.01)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))
    adam = optimizers.Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())



    checkpointer = ModelCheckpoint(save_model_dir_path,
                                   monitor='val_loss', verbose=1, save_best_only=True)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=1600/(bath_size/32),
        epochs=num_epoch,
        validation_data=val_generator,
        validation_steps=2000,
        callbacks=[checkpointer]
    )
    model_json = model.to_json()
    with open("../model_weights/model_sw_3.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../model_weights/model_sw_3.h5")
    print("Saved model to disk")

    scores = model.evaluate_generator(test_generator, nb_test_samples // bath_size)
    print("Test accuracy: %.2f%%" % (scores[1]*100))

    val_scores = model.evaluate_generator(val_generator, nb_val_samples // bath_size)
    print("Val accuracy: %.2f%%" % (val_scores[1]*100))
    #scores = model.evaluate(test_data, test_labels, verbose=0)
    #print("Точность работы на тестовых данных: %.2f%%" % (scores[1]*100))
    train_scores = model.evaluate_generator(train_generator, nb_train_samples // bath_size)
    print("Train accuracy: %.2f%%" % (train_scores[1]*100))
    plt.plot(history.history['acc'], label='Точность на обучающем наборе')
    plt.plot(history.history['val_acc'], label='Точность на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Точность')
    plt.legend()
    plt.savefig('plot2.png')
    plt.clf()


    plt.plot(history.history['loss'], label='Ошибка на обучающем наборе')
    plt.plot(history.history['val_loss'], label='Ошибка на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.savefig('plot3.png')

def test_vid():
    classifier = load_model("model_weights/model_sw_weights.hdf5")
    class_labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление', 'Нейтральный']

    cap = cv2.VideoCapture(0)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

    while True:
        # Capture frame-by-frame
        __, frame = cap.read()

        # Use MTCNN to detect faces
        result = detector.detect_faces(frame)
        if result != []:
            for person in result:
                # print(result)
                bounding_box = person['box']
                keypoints = person['keypoints']
                gray = frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[bounding_box[1] + 5:bounding_box[1] + bounding_box[3] - 5,
                           bounding_box[0] + 20:bounding_box[0] + bounding_box[2] - 20]
                '''roi_gray = gray[bounding_box[1]:bounding_box[1] + bounding_box[3],
                           bounding_box[0]:bounding_box[0] + bounding_box[2]]'''
                face = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = face.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                cv2.rectangle(frame,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + bounding_box[2] + 15, bounding_box[1] + bounding_box[3] + 15),
                              (245, 135, 66),
                              2)
                cv2.rectangle(frame,
                              (bounding_box[0], bounding_box[1]),
                              (bounding_box[0] + ((bounding_box[3] - 10)), bounding_box[1] + 20),
                              (245, 135, 66),
                              -1)

                # cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
                # cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
                # cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 2)
                # cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
                # cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (bounding_box[0] + 10, bounding_box[1] + 15)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
        else:
            cv2.putText(frame, "Лицо не найдено", (20, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_img():
    classifier = load_model("model_weights/model_sw_weights.hdf5")
    class_labels = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Грусть', 'Удивление', 'Нейтральный']
    image = cv2.cvtColor(cv2.imread("test_img/best-funny-facts.jpg"),
                         cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[bounding_box[1] + 5:bounding_box[1] + bounding_box[3] - 5,
               bounding_box[0] + 20:bounding_box[0] + bounding_box[2] - 20]
    face = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    roi = face.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2] + 15, bounding_box[1] + bounding_box[3] + 15),
                  (245, 135, 66),
                  2)
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + ((bounding_box[3] // 3)), bounding_box[1] + 20),
                  (245, 135, 66),
                  -1)

    #cv2.circle(image,(keypoints['left_eye']), 3, (255,0,0), 2)
    #cv2.circle(image,(keypoints['right_eye']), 3, (255,0,0), 2)
    #cv2.circle(image,(keypoints['nose']), 2, (255,0,0), 2)
    #cv2.circle(image,(keypoints['mouth_left']), 2, (255,0,0), 2)
    #cv2.circle(image,(keypoints['mouth_right']), 2, (255,0,0), 2)
    preds = classifier.predict(roi)[0]
    label = class_labels[preds.argmax()]
    label_position = (bounding_box[0] + 10, bounding_box[1] + 15)
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite("9.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(result)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, help='Выбор режима работы программы test_vid, test_img или train')
args = parser.parse_args()
if args.mode == 'train':
    train()
elif args.mode == 'test_vid':
    test_vid()
elif args.mode == 'test_img':
    test_img()
