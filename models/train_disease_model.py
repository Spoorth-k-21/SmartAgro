import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Path to your dataset
data_dir = "../data/PlantVillage"

# Data preparation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Save model

model.save("disease_model.h5")
print("✅ Model saved as 'models/disease_model.h5'")

# Save class labels
import json
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("✅ Class labels saved as 'models/class_indices.json'")
print("Current working directory:", os.getcwd())
print("Dataset path exists:", os.path.exists(data_dir))
print("Folders inside dataset:", os.listdir(data_dir))