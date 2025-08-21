import cv2
import numpy as np

def compute_hog(image, K, D):
    """
    Compute the Histogram of Oriented Gradients (HOG) feature vector for a given grayscale image.
    
    Parameters:
        image: Grayscale input image
        K: Number of orientation bins (e.g., 4 or 8)
        D: Block size (grid dimension) in pixels (e.g., 16)
    
    Returns:
        Normalized HOG feature vector
    """
    # Compute gradients using Sobel filters
    # to get rate of change in the pixel
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)
    
    # Compute magnitude and orientation
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * (180 / np.pi)  # Convert to degrees
    orientation = np.mod(orientation + 360, 180)  # Ensure orientation is in [0, 180)
    
    # Get image dimensions
    h, w = image.shape
    
    # Number of blocks
    M = h // D
    N = w // D
    
    # Initialize feature vector
    feature_vector = []
    
    # Process each block
    for i in range(M):
        for j in range(N):
            # Extract block
            block_mag = magnitude[i*D:(i+1)*D, j*D:(j+1)*D]
            block_ori = orientation[i*D:(i+1)*D, j*D:(j+1)*D]
            
            # Compute histogram for the block
            hist, _ = np.histogram(block_ori, bins=K, range=(0, 180), weights=block_mag)
            feature_vector.extend(hist)
    
    # Convert to numpy array
    feature_vector = np.array(feature_vector)
    
    # Normalize the feature vector using L2 norm
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    
    return feature_vector

def compute_similarity(image1, image2, K, D):
    """
    Compute the cosine similarity between two images based on their HOG feature vectors.
    
    Parameters:
        image1: First grayscale input image
        image2: Second grayscale input image
        K: Number of orientation bins
        D: Block size in pixels
    
    Returns:
        Cosine similarity score (float)
    """
    # Compute HOG feature vectors for both images
    hog1 = compute_hog(image1, K, D)
    hog2 = compute_hog(image2, K, D)
    
    # Compute cosine similarity (dot product since vectors are normalized)
    similarity = np.dot(hog1, hog2)
    
    return similarity

# Example usage
if __name__ == "__main__":
    # Load images 
    #image1 = cv2.imread('/Users/razeek_j/Documents/SRH Docs/Semester 1/Block 3/Image Processing/Task 2/cheeta.jpg', cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread('/Users/razeek_j/Documents/SRH Docs/Semester 1/Block 3/Image Processing/Task 2/jagur.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('/Users/razeek_j/Documents/SRH Docs/Semester 1/Block 3/Image Processing/Task 2/cheeta.jpg', cv2.IMREAD_GRAYSCALE)
    #image2 = cv2.imread('/Users/razeek_j/Documents/SRH Docs/Semester 1/Block 3/Image Processing/Task 2/jagur.jpg', cv2.IMREAD_GRAYSCALE)

    # Set parameters
    K = 8  # Number of orientation bins
    D = 16  # Block size
    
    # Compute similarity
    similarity = compute_similarity(image1, image2, K, D)
    print(f"Cosine similarity between the two images: {similarity}")