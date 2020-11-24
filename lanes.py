# This code will be used as the main material to build a simple application 
# that can detect vehicle lanes

import cv2 #Using OpenCV library
import numpy as np #Using Numpy library
import matplotlib.pyplot as plt

#Method ini dibuat untuk melakukan beberapa pemrosesan gambar, mulai dari grayscaling, smoothing hingga pada edge detection
def imageProcessing (image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #define object to initiate color grading process from RGB to Grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0) #Define object yang berisi fungsi GaussianBlur untuk melakukan proses blurring pada image
    cannyImage = cv2.Canny(blur, 50, 150) #Define object yang berisi method untuk melakukan edge detection dengan mencari piksel yang memiliki gradasi tertinggi
    return cannyImage

#Define method untuk mencari titik koordinat garis lane
def create_coordinates(image, line_parameters): 
    slope, intercept = line_parameters 
    y1 = image.shape[0] 
    y2 = int(y1 * (3 / 5)) 
    x1 = int((y1 - intercept) / slope) 
    x2 = int((y2 - intercept) / slope) 
    return np.array([x1, y1, x2, y2]) 

def average_slope_intercept(image, lines): 
    left_fit = [] 
    right_fit = [] 
    for line in lines: 
        x1, y1, x2, y2 = line.reshape(4) 
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  
        slope = parameters[0] 
        intercept = parameters[1] 
        if slope < 0: 
            left_fit.append((slope, intercept)) 
        else: 
            right_fit.append((slope, intercept))    
    left_fit_average = np.average(left_fit, axis = 0) 
    right_fit_average = np.average(right_fit, axis = 0) 
    left_line = create_coordinates(image, left_fit_average) 
    right_line = create_coordinates(image, right_fit_average) 
    return np.array([left_line, right_line]) 

#Method ini digunakan untuk melakukan kalkulasi titik potong yang akan dijadikan acuan dalam pembuatan garis
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, x2, y1, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

#Method ini dibuat untuk melakukan lokalisasi area yang ditandai sebagai lajur kendaraan
def region_of_interest (image):
    height = image.shape[0] #Objek yang mendeskripsikan pointer koordinat untuk mengacu pada titik paling atas gambar (Titik X)
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ]) #Menandai area pada gambar yang akan dilakukan lokalisasi objek
    mask = np.zeros_like(image) #Melakukan masking pada area gambar
    cv2.fillPoly(mask, polygons, 255) #Menegaskan koordinat gambar yang akan dilakukan masking
    masked_image = cv2.bitwise_and(image, mask) #Melakukan bitwise operation antara hasil output canny image dengan hasil masking image
    return masked_image

# cv2.imshow('hasil', gray) #Display a result image processing menggunakan library Opencv
# cv2.waitKey(0) #set delay untuk display hasil gambar, direkomendasikan value nya 0 supaya tampil secara kontinu (satuan nya milisecond)

image = cv2.imread('test_image.jpg') #Import image so that be process by OpenCv
lane_image = np.copy(image) #Make a copy of "image" so that cannot be affect the test_image file
canny = imageProcessing(lane_image) #menampilkan hasil gambar dari fungsi imageProcessing
cropped_image = region_of_interest(canny)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=60, maxLineGap=5) #Define object yang berisi method fungsi HoughLines
line_image = display_lines(lane_image, lines)
cv2.imshow("image", line_image)
cv2.waitKey(0)
# print(image.shape) #Menampilkan properties image (height, width, banyaknya channel color)
