import json
import os
import string
import random 
import nltk
import numpy as np
import requests # Ajouter la bibliothèque requests pour faire des requêtes à l'API
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
import re
import datetime
import speech_recognition as sr



nltk.download("punkt")
nltk.download("wordnet")

# initialisation de lemmatizer pour obtenir la racine des mots
lemmatizer = WordNetLemmatizer()

# création des listes
words = []
classes = []
doc_X = []
doc_y = []

# parcourir avec une boucle For toutes les intentions
# tokéniser chaque pattern et ajouter les tokens à la liste words, les patterns et
# le tag associé à l'intention sont ajoutés aux listes correspondantes
data_directory = "data"  # Utilisez un chemin relatif au lieu d'un chemin absolu

all_intents_data = []

for file in os.listdir(data_directory):
    file_path = os.path.join(data_directory, file)

    if file.endswith(".json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
            all_intents_data.extend(data["intents"])
            

        # Parcourir avec une boucle For toutes les intentions
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                tokens = nltk.word_tokenize(pattern)
                words.extend(tokens)
                doc_X.append(pattern)
                doc_y.append(intent["tag"])

            # Ajouter le tag aux classes s'il n'est pas déjà là
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

# lemmatiser tous les mots du vocabulaire et les convertir en minuscule
# si les mots n'apparaissent pas dans la ponctuation
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

# trier le vocabulaire et les classes par ordre alphabétique et prendre le
# set pour s'assurer qu'il n'y a pas de doublons
words = sorted(set(words))
classes = sorted(set(classes))


# liste pour les données d'entraînement
training = []
out_empty = [0] * len(classes)

# création du modèle d'ensemble de mots
for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    # marque l'index de la classe à laquelle le pattern atguel est associé à
    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    # ajoute le one hot encoded BoW et les classes associées à la liste training
    training.append([bow, output_row])

# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)

# séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# définition de quelques paramètres
input_shape = (len(train_X[0]),)
output_shape = len(train_y[0])
epochs = 500

# modèle Deep Learning
model = Sequential()
model.add(Dense(512, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate=0.01,
decay_steps=1000,
decay_rate=0.9
)
adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

model.fit(x=train_X, y=train_y, epochs=epochs, verbose=1)

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("VOCAL MODE ACTIVATED : ")
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio, language="fr-FR")
        return text
    except sr.UnknownValueError:
        return "ERROR"
    except sr.RequestError as e:
        return "ERROR"

#Fonction pour nettoyer le texte
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

#Fonction pour créer un sac de mots
def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)

#Fonction pour prédire la classe à partir d'un texte
def pred_class(text, vocab, labels):
    bow = bag_of_words(text, vocab)
    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def extract_year(text):
    match = re.search(r"\b\d{4}\b", text)
    if match:
        return int(match.group())
    else:
        currentDateTime = datetime.datetime.now()
        date = currentDateTime.date()

        return date.year

#Fonction pour obtenir la réponse à partir des intentions prédites
def get_response(intents_list, all_intents_data, year=None):
    tag = intents_list[0]
    for i in all_intents_data:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
        if tag == "driver_standings":
            # Récupérer les noms de famille des pilotes de la saison {year} à partir de l'API
            response = requests.get(f"http://ergast.com/api/f1/{year}/drivers.json")
            data = response.json()
            drivers = data["MRData"]["DriverTable"]["Drivers"]
            family_names = [driver["familyName"] for driver in drivers]
            # Remplacer les "..." par les noms de famille des pilotes dans la réponse renvoyée
            result = result.replace("...", ", ".join(family_names))
            # Remplacer "/year/" par l'année
            result = result.replace("/year/", str(year))
            break
    return result

#lancement du chatbot
while True:
    message = input("YOU :")
    if message == "vocal":
        message = get_audio()
        print(message)
    if message == "FIN":
        print("FERMETURE")
        break
    else:
        try:
            intents = pred_class(message, words, classes)
            yearSuggested = extract_year(message)
            result = get_response(intents, all_intents_data, yearSuggested) # Ajouter l'année pour l'API
            print(result)
        except:
            print("Votre requête manque de précison, ou alors je n'ai pas compris")
