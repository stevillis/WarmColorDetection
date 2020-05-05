import cv2
import numpy as np
import pytesseract
from PIL import Image

src_path = 'C:/Users/stevi/Documents/StackOverflow/WarmColorDetection/'
pytesseract.pytesseract.tesseract_cmd = 'C:/Users/stevi/AppData/Local/Tesseract-OCR/tesseract.exe'


def find_temperature_range(img, y1=0, y2=0, x1=0, x2=0):
    '''
    Find the number that indicates the temperature range for that image.
    :param img: The image where the temperature range is located.
    :param y1: Start of the temperature scale label height.
    :param y2: End of the temperature scale label height.
    :param x1: Start of of the temperature scale label width.
    :param x2: End of of the temperature scale label width.
    :return: A temperature range value read.
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]  # ROI - Region of Interest

    thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = np.ones((1, 1), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Recognize text with tesseract for python
    binimagem = Image.fromarray(dilation)
    temperature_range = pytesseract.image_to_string(binimagem,
                                                    config='--psm 10 -c tessedit_char_whitelist=01234567890.')
    return float(temperature_range)


def find_warm_pixel(img, radius=3):
    '''
    Find warm pixel in the given image
    :param img: Image where the warm pixel will be searched
    :param radius: kernel
    :return: A tuple with the values of (minVal, maxVal, minLoc, maxLoc)
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian Blur to the image then find the brightest region
    gray = cv2.GaussianBlur(gray, (radius, radius), 0)
    return cv2.minMaxLoc(gray)


if __name__ == '__main__':
    # Loop over all images and show the warm point of all of them
    for i in range(1, 6):
        img = cv2.imread(f'img/img{i}.jpg', 1)
        y, x, _ = img.shape
        img_copy = img.copy()

        max_temp_range = find_temperature_range(img_copy, 45, 60, 280, 315)
        min_temp_range = find_temperature_range(img_copy, 178, 194, 280, 315)

        if i == 1:
            max_temp_range = 97.7  # Could not read the correct number only for this case, as it's showing 77

        (minVal, maxVal, minLoc, maxLoc) = find_warm_pixel(img_copy)

        # Converting a pixel value based on minimum and maximum value range read from the image
        # new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
        old_value = maxVal
        old_min = 0
        old_max = 255
        temperature = ((old_value - old_min) / (old_max - old_min)) * (max_temp_range - min_temp_range) + min_temp_range

        circle_radius = 3
        cv2.circle(img, maxLoc, circle_radius, (255, 0, 0), 2)  # draw a circle around the britest pixel
        cv2.putText(img, f'Coordinate: {maxLoc}', (122, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img, f'Value: {temperature:.2f}', (122, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (255, 255, 255), 1,
                    cv2.LINE_AA)

        # Display the result
        cv2.namedWindow(f'Image {i}', cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(f'Image {i}', x, y)
        cv2.imshow(f'Image {i}', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
