import cv2

def image_edge_detection(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (1, 1), 1)

    # Detect edges using the Sobel operator
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=1)

    # Combine the results
    edges_sobel = cv2.bitwise_or(sobel_x, sobel_y)
    edges_sobel = cv2.bitwise_not(edges_sobel)

    # Normalize the image to 0-255 and convert to uint8
    normalized_edges = cv2.normalize(edges_sobel, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imshow("kk" , normalized_edges)
    cv2.waitKey(0)
    # Define the output filename
    output_filename = image_path.split('/')[-1]  # Modify this as needed
    cv2.imwrite(output_filename, edges_sobel)

    return edges_sobel

image_edge_detection('../dataset/train/earth/001.jpg')