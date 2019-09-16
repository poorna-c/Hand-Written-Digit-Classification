import numpy as np
import matplotlib.pyplot as plt
import pickle
import pygame
from pygame.locals import MOUSEBUTTONDOWN,QUIT,MOUSEMOTION,MOUSEBUTTONUP
from PIL import Image
import cv2
# from keras.models import load_model

pygame.init()

WHITE = (255,255,255)
BLACK = (0,0,0)
drawing = False
predict = False
model = pickle.load(open('mnist.pkl','rb'))
# model = load_model('MNIST.h5')
screen = pygame.display.set_mode((300, 300), 0, 32)
screen.fill(WHITE)
pygame.display.set_caption("Draw Digit")
def preprocess_img(img):
    #img = abs(img-255)
    img = cv2.bitwise_not(img)
    img = img/255
    img = img.reshape(1,28,28,1)
    return img

def predict_and_plot():

    plt.close('all')
    pygame_screen = pygame.surfarray.array3d(screen)
    PIL_image = Image.fromarray(pygame_screen)
    PIL_image_transpose = PIL_image.transpose(Image.TRANSPOSE)
    PIL_image_28x28 = PIL_image_transpose.resize((28,28), Image.ANTIALIAS)
    image_array_3d = np.array(PIL_image_28x28)
    image_array_2d = image_array_3d[:,:,0]
    image_array = preprocess_img(image_array_2d)
    prediction = model.predict_proba(image_array)
    _, ax = plt.subplots(1,2,figsize=(8,4))
    ax[0].bar([0,1,2,3,4,5,6,7,8,9],prediction[0]*100)
    ax[0].set_xticks([0,1,2,3,4,5,6,7,8,9])
    ax[0].set_yticks([10,20,30,40,50,60,70,80,90,100])
    ax[0].set_title("Prediction: "+str(np.argmax(prediction[0]))+"\nAccuracy: "+str(np.round(prediction[0][np.argmax(prediction[0])]*100,2)))
    ax[1].imshow(image_array.reshape(28,28),cmap='binary')
    plt.tight_layout()
    plt.show(block=False)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()
        elif event.type == MOUSEMOTION:
            if (drawing):
                mouse_position = pygame.mouse.get_pos()
                pygame.draw.circle(screen, BLACK, mouse_position, 8)
        elif event.type == MOUSEBUTTONUP and event.button==1:
            drawing = False
            predict = True
        elif event.type == MOUSEBUTTONDOWN and event.button==1:
            drawing = True
        elif event.type == MOUSEBUTTONDOWN and event.button == 3:
            screen.fill(WHITE)
        if predict:
            predict = False
            predict_and_plot()

    pygame.display.update()
