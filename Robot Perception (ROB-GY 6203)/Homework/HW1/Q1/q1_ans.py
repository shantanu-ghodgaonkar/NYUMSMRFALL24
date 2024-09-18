import cv2
from pathlib import Path

if __name__ == "__main__":
    # Resolve image path
    img_path = Path('Q1/for_watson.png').absolute()
    # Read the given image and store it in a variable
    img = cv2.imread(img_path.__str__())
    # Convert image from BGR to Grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding with kernel size of 25 and C = 0
    img_decoded = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 0)
    # Display decoded image
    cv2.imshow('Decoded Image', img_decoded)
    # Wait for keyboard interrupt
    cv2.waitKey()
    # closing all open windows
    cv2.destroyAllWindows()
    # Store image in folder
    cv2.imwrite((img_path.parent.__str__() +
                '/for_watson_decoded.png'), img_decoded)
