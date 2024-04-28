import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_signatures(img1, img2):
    # Convert the images to grayscale
        # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSI)
    similarity_index = ssim(gray1, gray2)
    
    # Define a threshold for similarity
    similarity_threshold = 0.95
    print(similarity_index)


    # Check if the similarity index is above the threshold
    if similarity_index > similarity_threshold:
        return True
    else:
        return False
    

    # Load the original signature image
original_image = cv2.imread(r'D:\downloads\image1.jpg')
cv2.imshow('original_image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert the image to grayscale
gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale_image', gray_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Threshold to obtain binary image
_, binary_original = cv2.threshold(gray_original, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('grayscale_image', binary_original)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the signature contour from the original image
contours1, _ = cv2.findContours(binary_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour1 = contours1[0]

# Create a mask for the original signature
mask = np.zeros(gray_original.shape, dtype=np.uint8)
cv2.drawContours(mask, [contour1], 0, 255, -1)

# Extract the signature region from the original image
original_signature = cv2.bitwise_and(original_image, original_image, mask=mask)
cv2.imshow('grayscale_image', original_signature)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Load the input signature image
input_image1 = cv2.imread(r'D:\downloads\image1.jpg')
input_image = cv2.resize(input_image1, (original_image.shape[1], original_image.shape[0]),interpolation=cv2.INTER_AREA)
# Convert the image to grayscale
gray_input = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('grayscale_image', gray_input)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Threshold to obtain binary image
_, binary_input = cv2.threshold(gray_input, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('grayscale_image', binary_input)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Extract the signature contour from the input image
contours2, _ = cv2.findContours(binary_input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour2 = contours2[0]

# Create a mask for the input signature
mask = np.zeros(gray_input.shape, dtype=np.uint8)
cv2.drawContours(mask, [contour2], 0, 255, -1)

# Extract the signature region from the input image
input_signature = cv2.bitwise_and(input_image, input_image, mask=mask)
cv2.imshow('grayscale_image', input_signature)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compare the signatures
if compare_signatures(original_signature, input_signature):
    print('The signatures are matched')
else:
    print('The signatures are not Matched.')