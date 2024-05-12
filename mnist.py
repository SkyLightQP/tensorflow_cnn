import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# 랜덤값 시드 고정. 매번 동일한 랜덤값을 반환합니다.
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.2, shuffle=True, stratify=y_train)

print(f"X Train Shape: {x_train.shape}")

x_train = x_train.astype('float32') / 255
x_valid = x_valid.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = Sequential([
    Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dropout(.25),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, validation_data=(x_valid, y_valid), epochs=10)

model.evaluate(x_test, y_test, verbose=2)

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.axis('off')
    predictions = model.predict(x_test[i].reshape(1, 28, 28, 1))
    score = tf.nn.softmax(predictions[0])
    # argmax()는 배열 중 가장 값이 큰 요소의 인덱스를 반환한다.
    plt.title(f'answer: {np.argmax(score)}')

plt.show()
