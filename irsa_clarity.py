import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from PIL import Image, ImageEnhance
import PySimpleGUI as sg



def change_function(event):

    contrast = ImageEnhance.Contrast(image_file)
    contrast.enhance(freq_slider.val).save('new_image.jpg')
    img = cv2.imread('new_image.jpg')

    numpydata = np.asarray(img)

    numpydata[mask1] = data_template[mask1]
    numpydata[mask2] = data_template[mask2]
    cv2.imwrite('new_image.jpg', numpydata)


    image_file2 = Image.open('new_image.jpg')
    enhancer = ImageEnhance.Brightness(image_file2)
    img_output = enhancer.enhance(freq_slider.val)
    img_output.save('new_image.jpg')
    img = cv2.imread('new_image.jpg')

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    width = 1500
    height = img.shape[0] - 200  # keep original height
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # showing image
    cv2.imshow("changed image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mask_number = sg.ELEM_TYPE_INPUT_SPIN
    # get data
    data = np.load('x_test_truck_irsa.npy')


    # creating image
    cv2.imwrite('test.jpg', data)

    # save image
    image_path = "test.jpg"
    img = cv2.imread('test.jpg')

    data = np.asarray(img)
    mask1 = data<70
    mask2 = data>180
    data_template = np.copy(data)


    # open image
    image_file = Image.open(image_path)

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    plt.imshow(img)

    # show original image
    # cv2.imshow("original image", image_file)

    # creating slider
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    init = 1

    freq_slider = Slider(
        ax=axfreq,
        label='clarity',
        valmin=0.1,
        valmax=30,
        valinit=init,
    )
    change = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(change, 'change', hovercolor='0.975')

    # add action listener
    button.on_clicked(change_function)

    # showing slider
    plt.show()
