import numpy as np
import cv2
from itertools import chain
from faceless import *

'''
Facial Recognition using PCA
'''

# Constants
IMG_HEIGHT = 112
IMG_WIDTH = 92
AMOUNT_EIGENFACES = 10
#Lista de paths donde estan las personas, se usa el path como el nombre de la persona
persons = list(map(lambda x: "../faces/s%d" %(x),range(1,41)))
persons.append("../faces/pedro") #comenten esta linea para agregar o no a pedro
#lista de fotos usada para el training data, usa todas las fotos de las perona en "persons"
photos_per_person = [8,7];

training_db, training_classes = load_images(persons, photos_per_person)
print training_classes
mean_face = calculate_mean_face(training_db)

print_image(np.reshape(mean_face, [IMG_HEIGHT, IMG_WIDTH]), "mean", 500)

#Calculate normalized database
dbc = training_db - mean_face
#Find eigenfaces of the database
eigenfaces = calc_eigenfaces(dbc,AMOUNT_EIGENFACES)

face_to_find,target_class = load_images(persons,range(1,11))

win = 0
total = 0
for index in range(0,face_to_find.shape[0]):
    results =  find_closest_faces(face_to_find[index]-mean_face,eigenfaces,dbc)
    person_proba = analize_matchs(results, training_classes)

    tag = ("Target '%s' Probabilites %s") %(target_class[index],person_proba[0:3])
    total = total+1
    if person_proba[0][0] == target_class[index]:
        # print "GANO: " + tag
        win = win +1;
    else:
        print "PERDIO: " + tag

print "RATIO: %f, WIN: %d, TOTAL: %d, LOST: %d" %(100.0*win/total,win,total,total-win)
cv2.waitKey(0)
cv2.destroyAllWindows()

