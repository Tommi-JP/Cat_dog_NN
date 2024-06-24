### Project Title: Cat and Dog Image Classification

This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on a dataset of cat and dog images, with data augmentation applied to enhance the training process.

#### Table of Contents

- [Installation](#installation)
- [Usage](#usage)

#### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Tommi-JP/Cat_dog_NN
    cd Cat_dog_NN
    ```
2. Create a virtual environment and activate it (optional):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Install Git LFS:
    - macOS:
      ```bash
      brew install git-lfs
      ```
    - Windows: Download and Install [Git LFS for Windows](https://git-lfs.github.com/).
    - Linux: Follow the Instructions [Git LFS:n dokumentation](https://github.com/git-lfs/git-lfs/wiki/Installation).

5. Initialize Git LFS in your project:
    ```bash
    git lfs install
    ```

6. Download dataset:
    ```bash
    git lfs pull
    ```

#### Usage

1. Prepare your dataset:
    - Ensure your dataset directory structure is as follows:
      ```
      Cat_Dog_data/
      ├── train/
      │   ├── cat/
      │   └── dog/
      └── test/
          ├── cat/
          └── dog/
      ```

2. Run the training script:
    ```bash
    python main.py
    ```