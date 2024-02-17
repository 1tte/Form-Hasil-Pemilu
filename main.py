import cv2
import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print("Image not loaded.")
        return

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw bounding box around the largest contour
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Divide the bounding box into three sections vertically
        section_height = h // 3
        for i in range(3):
            section_y = y + i * section_height
            section = image[section_y:section_y+section_height, x:x+w]

            # Save each section as an individual image
            cv2.imwrite(f'section_{i+1}.png', section)

            # Display each section
            cv2.imshow(f'Section {i+1}', section)

        # Filter contours based on area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        # Draw bounding boxes around filtered contours
        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the original image with bounding boxes
        cv2.imshow('Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No contours found in the image.")

# Specify the image path
image_path = 'sample.png'

# Process the image
process_image(image_path)
