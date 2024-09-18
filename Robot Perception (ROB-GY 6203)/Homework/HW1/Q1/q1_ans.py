import cv2  # Import OpenCV for image processing operations
from pathlib import Path  # Import Path for file path resolution

if __name__ == "__main__":
    # The above condition ensures that the code inside this block runs only when the script is executed directly,
    # not when it's imported as a module in another script.

    # Obtain the absolute path of the image file
    img_path = Path('Q1/for_watson.png').absolute()
    # Read the given image and store it in a variable
    img = cv2.imread(img_path.__str__())
    # Convert image from BGR to Grayscale, which is required for optimising thresholding
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive mean thresholding to the gray image. Here, for each pixel, the
    # algorithm calculates the mean of the pixel values in a 25x25 neighborhood centered around
    # that pixel. It then subtracts the constant C (which is 0 in this case) from this mean value
    # to compute the threshold for that pixel. If the pixel's intensity is greater than this
    # threshold, it's set to 255 (white); otherwise, it's set to 0 (black).
    img_decoded = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 0)
    # Display decoded image
    cv2.imshow('Decoded Image', img_decoded)
    # Wait for keyboard interrupt, i.e., keep the image window open until a key is pressed
    cv2.waitKey()
    # close all open windows
    cv2.destroyAllWindows()
    # Store image to a file in the same folder as the original image
    cv2.imwrite((img_path.parent.__str__() +
                '/for_watson_decoded.png'), img_decoded)
