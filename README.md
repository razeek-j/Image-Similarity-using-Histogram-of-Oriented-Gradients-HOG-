# Image Similarity using Histogram of Oriented Gradients (HOG)

This repository contains a Python script that implements the Histogram of Oriented Gradients (HOG) algorithm to create a feature vector, or "fingerprint," for images. It then uses these fingerprints to calculate the structural similarity between two images using cosine similarity.

This technique is a classic and powerful method in computer vision, often used as a feature descriptor for tasks like **object detection**, **image matching**, and **content-based image retrieval**.

---

## 🔬 How the Code Works

The core idea is to describe the structure and shape of objects in an image by analyzing the distribution of gradient orientations (the direction of intensity changes). An image with a similar structure (e.g., two different photos of a cheetah) will have a similar HOG fingerprint, even if the lighting or colors are different.

The process is broken down into two main functions:

### 1. `compute_hog()` - Creating the Image Fingerprint
This function takes an image and generates its HOG feature vector. 

The steps are:
1.  **Gradient Calculation:** The code first calculates the horizontal ($`G_x`$) and vertical ($`G_y`$) gradients of the image using **Sobel filters**. This step identifies the intensity changes at every pixel, effectively highlighting edges and textures.
2.  **Magnitude and Orientation:** From the gradients, two values are computed for each pixel:
    * **Magnitude:** The "strength" of the edge ($`\sqrt{G_x^2 + G_y^2}`$). Strong edges (like a sharp outline) have high magnitudes.
    * **Orientation:** The direction of the edge (e.g., vertical, horizontal, or diagonal), calculated in degrees.
3.  **Block Histograms:** The image is divided into a grid of blocks (e.g., 16x16 pixels, defined by `D`). For each block, a histogram is created. This histogram has `K` bins, each representing a range of orientations (e.g., 0-22.5°, 22.5-45°, etc.). The algorithm goes through each pixel in the block and adds its magnitude to the histogram bin corresponding to its orientation. This builds a summary of the dominant edge directions within that block.
4.  **Normalization:** The final feature vector (a long list containing all block histograms) is normalized using the **L2 norm**. This crucial step makes the descriptor robust against changes in lighting and contrast. A brightly lit cheetah and a dimly lit one will produce a very similar normalized HOG vector.

### 2. `compute_similarity()` - Comparing the Fingerprints
This function compares the two HOG vectors using **Cosine Similarity**.

* **What it is:** Cosine similarity measures the cosine of the angle between two vectors. It's a way to determine how "similarly pointed" they are.
* **The Result:**
    * A score of **1.0** means the vectors are identical (the images are structurally the same).
    * A score of **0.0** means the vectors are completely different.
    * A score in between represents the degree of similarity.

---

## ⚙️ Packages Used

The script relies on two core scientific computing packages in Python:

* **OpenCV (`cv2`):** A powerful, open-source computer vision library.
    * **Why it was chosen:** It provides highly optimized, pre-built functions for common image processing tasks.
    * **What it does here:**
        * `cv2.imread()`: Loads the images from your disk into a format we can work with.
        * `cv2.Sobel()`: An efficient and standard function for calculating the image gradients needed for HOG.

* **NumPy (`numpy`):** The fundamental package for numerical operations in Python.
    * **Why it was chosen:** It's essential for handling the arrays and mathematical calculations required by the HOG algorithm.
    * **What it does here:**
        * Performs array operations and mathematical calculations (e.g., `np.sqrt`, `np.arctan2`).
        * `np.histogram()`: Efficiently computes the histogram for each block.
        * `np.linalg.norm()` and `np.dot()`: Used for vector normalization and the final cosine similarity calculation.

---

## 🚀 How to Use

1.  **Install Dependencies:**
    Make sure you have Python, OpenCV, and NumPy installed.
    ```bash
    pip install opencv-python numpy
    ```

2.  **Set Image Paths:**
    In the `if __name__ == "__main__":` block at the bottom of the script, change the file paths in the `cv2.imread()` functions to point to your two images.
    ```python
    # Load images
    image1 = cv2.imread('/path/to/your/first_image.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('/path/to/your/second_image.jpg', cv2.IMREAD_GRAYSCALE)
    ```

3.  **Run the Script:**
    Execute the Python file from your terminal.
    ```bash
    python your_script_name.py
    ```

4.  **Tune Parameters (Optional):**
    * `K`: Number of orientation bins. A higher number captures more detail about orientation but creates a larger feature vector. `8` is a common value.
    * `D`: The size of the blocks in pixels. This should be chosen based on the scale of the features you want to capture. `16` is a good starting point.

---

## 📊 Results

Here you can display the images you are comparing and the resulting similarity score.

### Image 1: Jaguar

![Image 1](./jaguar.jpg)

### Image 2: Cheetah

![Image 2](./cheetah.jpg)

### Similarity Score

The script will print the cosine similarity to the console.

**Result:**
