# Sign Language Detection using OpenCV

## Project Overview

This repository hosts a real-time sign language detection system developed using OpenCV. The project aims to bridge communication gaps by enabling computers to interpret sign language gestures captured via a webcam. It involves stages of data collection, dataset preprocessing, and training a classifier to accurately recognize various signs.

## Features

-   **Real-time Detection:** Utilizes webcam feed for live sign language recognition.
-   **Custom Dataset Creation:** Includes scripts to collect and build a personalized dataset of sign gestures.
-   **Robust Preprocessing:** Employs techniques to prepare image data for optimal model training.
-   **Classifier Training:** Trains a machine learning model to classify different sign gestures.
-   **Modular Design:** Separated functionalities for data collection, processing, training, and classification.

## Project Structure

-   `Dataset/`: This directory will store the images collected for each sign language gesture. Each sign will typically have its own subdirectory within this folder.
-   `__pycache__/`: (Automatically generated) Contains compiled Python bytecode files.
-   `Classifier.py`: Contains the main logic for the real-time sign language detection. This script will load the trained model and use the webcam feed to predict and display the detected sign.
-   `Data_collection.py`: A script used to capture images from a webcam and organize them into the `Dataset/` directory. This is crucial for building your own custom sign language dataset.
-   `Dataset_processing.py`: Handles the preprocessing of the collected image data. This may include resizing, normalization, feature extraction, or other transformations necessary before training the classifier.
-   `Train_test_classifier.py`: This script is responsible for training the sign language classification model using the preprocessed dataset. It also includes functionalities for evaluating the model's performance.

## Installation and Setup

To set up and run this project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/PrioAhmed19/Sign-Language-Detection-using-Open-CV.git
    cd Sign-Language-Detection-using-Open-CV
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install required libraries:**
    This project primarily uses OpenCV and potentially other libraries for machine learning (e.g., scikit-learn, TensorFlow, Keras). You can install them using pip:
    ```bash
    pip install opencv-python numpy scikit-learn
    # Add other libraries if necessary based on the specific implementation in the scripts
    ```

## Usage

### 1. Data Collection

Run `Data_collection.py` to capture images for your sign language gestures. Follow the on-screen instructions to create folders for each sign and collect a sufficient number of samples.

```bash
python Data_collection.py
```

### 2. Dataset Processing

After collecting data, run `Dataset_processing.py` to preprocess the images. This step prepares your dataset for training.

```bash
python Dataset_processing.py
```

### 3. Train and Test Classifier

Execute `Train_test_classifier.py` to train your sign language detection model. This script will also evaluate the model's performance.

```bash
python Train_test_classifier.py
```

### 4. Real-time Detection

Once the model is trained, run `Classifier.py` to start the real-time sign language detection using your webcam.

```bash
python Classifier.py
```

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to fork the repository, create a new branch, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.


## Contact

For any questions or inquiries, please contact [PrioAhmed19](https://github.com/PrioAhmed19).


