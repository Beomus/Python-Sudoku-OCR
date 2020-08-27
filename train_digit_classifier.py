from source.model import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import os

if not os.path.isdir("output"):
    os.makedirs("output")

MODEL_PATH = "output/digit_classifier.h5"

INIT_LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 128

# grab the dataset
print("[INFO] accessing dataset...")
((x_train, y_train), (x_test, y_test)) = mnist.load_data()

# add a channel dimension to the digits
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# scale data to range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# convert the labels from integers to vectors
le = LabelBinarizer()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# initialize the optimizer and the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] Model compiled successfully.")

# train the model
print("[INFO] training the model...")
H = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1
)
print("[INFO] Model training completed.")

# evaluate the model
print("[INFO] evaluating the model")
predictions = model.predict(x_test)
print(classification_report(
    y_test.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in le.classes_]
))

# serialize the model to disk
print("[INFO] serializing the model...")
model.save(MODEL_PATH, save_format="h5")
