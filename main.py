import cv2
import numpy as np

image_path = 'sample.png'
image = cv2.imread(image_path)

if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        section_height = h // 3
        for i in range(3):
            section_y = y + i * section_height
            section = image[section_y:section_y+section_height, x:x+w]

            cv2.imwrite(f'section_{i+1}.png', section)

            cv2.imshow(f'Section {i+1}', section)

        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        for cnt in filtered_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No contours found in the image.")
else:
    print("Image not loaded.")
