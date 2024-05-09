import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2)

model = Sequential([
    # 합성곱 + 폴링 레이어
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2)),

    # 1차원으로 변환
    Flatten(),
    Dense(128, activation='relu'),
    # 특성이 10개이므로 10개 노드 사용
    Dense(10, activation='softmax')
])

# model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, validation_data=(x_val, y_val), epochs=20)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.axis('off')
    predictions = model.predict(x_test[i].reshape(1, 32, 32, 3))
    score = tf.nn.softmax(predictions[0])
    plt.title(f'{class_names[np.argmax(score)]} {100 * np.max(score):.2f}%')
plt.tight_layout()
plt.show()

model.evaluate(x_test, y_test, verbose=2)
