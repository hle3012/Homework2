import numpy as np
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential

# ADD
train_dir = "/content/drive/MyDrive/AI/Ảnh selfie cho chiều t2 18 5"
img_width, img_height = 200,200
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode="nearest")

# READ
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical")
num_classes = train_generator.num_classes
model=Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")])


# TRAIN
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_generator, epochs=100)

# PRED
import numpy as np
import matplotlib.pyplot as plt
path = "/content/drive/MyDrive/AI/Nguyen Huynh Le/resized_IMG_5492.jpeg"

img = load_img(path, target_size=(200,200))
plt.imshow(img)
plt.show()
img = np.array(img)
img = img/255.0
img = img.reshape(1,200,200,3)
pred = np.argmax(model.predict(img))

class_labels = {v: k for k, v in train_generator.class_indices.items()}
name = class_labels[pred]
print(f"Dự đoán: {name}")
