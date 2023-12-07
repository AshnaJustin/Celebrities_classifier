import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

image_dir=r"C:\Users\Ashna Justin\Downloads\Dataset_Celebrities\cropped"
lionel_messi=os.listdir(image_dir+ '\\lionel_messi')
maria_sharapova=os.listdir(image_dir+ '\\maria_sharapova')
roger_federer=os.listdir(image_dir+ '\\roger_federer')
serena_williams=os.listdir(image_dir+ '\\serena_williams')
virat_kohli=os.listdir(image_dir+ '\\virat_kohli')

print("--------------------------------------\n")

print('The length of lionel_messi images is',len(lionel_messi))
print('The length of maria_sharapova images is',len(maria_sharapova))
print('The length of roger_federer images is',len(roger_federer))
print('The length of serena_williams images is',len(serena_williams))
print('The length of virat_kohli images is',len(virat_kohli))



print("--------------------------------------\n")
            
dataset=[]
label=[]
img_siz=(128,128)


for i ,image_name in tqdm(enumerate(lionel_messi),desc="lionel_messi"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/lionel_messi/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(maria_sharapova),desc="maria_sharapova"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/maria_sharapova/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)

for i ,image_name in tqdm(enumerate(roger_federer),desc="roger_federer"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/roger_federer/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)

for i ,image_name in tqdm(enumerate(serena_williams),desc="serena_williams"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/serena_williams/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(3)

for i , image_name in tqdm(enumerate(virat_kohli),desc="virat_kohli"):
    if(image_name.split('.')[1]=='png'):
        image=cv2.imread(image_dir+'/virat_kohli/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(4)
        
        
dataset=np.array(dataset)
label = np.array(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.2,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")


x_train=x_train.astype('float')/255.0
x_test=x_test.astype('float')/255.0



model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(5,activation='softmax')
])

#compile the model
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
# Training the Model
history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.3)

# Model Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report:\n', classification_report(y_test, y_pred))

# Load and preprocess a single image
def preprocess_single_image(image_path):
    img_size = (128, 128)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image

# Replace 'path_to_your_image.png' with the path to the image you want to predict
images = [r"C:\Users\Ashna Justin\OneDrive\Documents\Desktop\Dataset_celebrities\Data\virat_kohli\virat_kohli2.png",
r"C:\Users\Ashna Justin\OneDrive\Documents\Desktop\Dataset_celebrities\Data\serena_williams\serena_williams2.png",
r"C:\Users\Ashna Justin\OneDrive\Documents\Desktop\Dataset_celebrities\Data\roger_federer\roger_federer1.png",
r"C:\Users\Ashna Justin\OneDrive\Documents\Desktop\Dataset_celebrities\Data\maria_sharapova\maria_sharapova1.png",
r"C:\Users\Ashna Justin\OneDrive\Documents\Desktop\Dataset_celebrities\Data\lionel_messi\lionel_messi1.png"]
                        

# Preprocess the single image
for i in images:
    single_image = preprocess_single_image(i)

    # Reshape the image to fit the model's input shape
    single_image = np.expand_dims(single_image, axis=0)

    # Make predictions using the model
    predictions = model.predict(single_image)
    predicted_class = np.argmax(predictions)
    class_names = ['lionel_messi', 'maria_sharapova', 'roger_federer', 'serena_williams', 'virat_kohli']
    predicted_label = class_names[predicted_class]

    print(f"The predicted label for the image is: {predicted_label}")



#We iterated through each player using for loops to check if the image is png, load it, convert it to RGB, resize it, and append it to the dataset with the corresponding label during the data processing phase.
#We transformed the dataset into Tensorflow-compatible numpy arrays as part of the dataset preparation process.
#A basic CNN model architecture was selected as the model. Compared to other models, it performs better on image classification tasks for which it was intended. 
#We set the batch size to 32 and the number of epochs to 50 during the training process. Additionally, a portion of the training data is reserved as the validation set because the validation split has been set to 0.3.

#We receive a classification report detailing the accuracy, precision, recall, and F1-score for each class during the model evaluation process. Our accuracy rate was 86.27%.
#Also, we have printed the predictions for every class.