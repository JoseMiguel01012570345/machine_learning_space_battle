import cv2
import os

def image_edge_detection(image_path, train=True , replacement=""):
    
    # Load the image
    try:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        return
    
    if image is None:
        return 
    
    image = cv2.resize( image , (1026,1026) )
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (1, 1), 1)

    # Detect edges using the Laplacian operator
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    # Normalize the image to 0-255 and convert to uint8
    normalized_laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Threshold the image to get binary edges
    _, binary_edges = cv2.threshold(normalized_laplacian, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    number_of_white = 0
    number_of_black = 0
    for i in range(len(binary_edges)):
        for j in range( len(binary_edges[i])):
            
            if binary_edges[i,j] > 128:
                binary_edges[i,j] = 255
                number_of_white += 1
            else:
                binary_edges[i,j] = 0
                number_of_black += 1
    
    if number_of_white < number_of_black:
        binary_edges = cv2.bitwise_not(binary_edges)
    
    # Define the output filename
    if train:
        print("./"+ replacement)
        cv2.imwrite("./" + replacement, binary_edges)
    else:
        print("./"+ replacement)
        cv2.imwrite("./" + replacement, binary_edges)

# Example usage

train_path = "../dataset/train/earth/"
val_path = "../dataset/validation/earth/"

len_train_path = len(os.listdir(train_path))
len_val_path = len(os.listdir(val_path))

# save the train dataset 
for i in range(len_train_path):
    
    if i < 9:
        image_edge_detection(f"{train_path}00{ i + 1 }.jpg" , replacement=f"{i}.jpg" )
    if i >= 10 and i < 99:
        image_edge_detection(f"{train_path}0{i + 1 }.jpg" , replacement=f"{i}.jpg" )
    if  i >= 100:
        image_edge_detection(f"{train_path}{i + 1}.jpg" , replacement=f"{i}.jpg" )


# save the validation dataset 
# for i in range(len_val_path):
#     image_edge_detection( f"{val_path}{ i + 641 }.jpg" , False , replacement=f"{i}.jpg" )
