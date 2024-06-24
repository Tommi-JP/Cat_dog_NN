# Cat_dog_NN
Neural network that recognizes images of cats and dogs

This project aims to classify images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on a dataset of cat and dog images, with data augmentation applied to enhance the training process.

Installation

    Clone the repository:

"""
git clone https://github.com/yourusername/cat-dog-classification.git
cd cat-dog-classification
"""

Create a virtual environment and activate it:

bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install the required packages:

bash

    pip install -r requirements.txt

Usage

    Prepare your dataset:
        Ensure your dataset directory structure is as follows:

        bash

    Cat_Dog_data/
    ├── train/
    │   ├── cat/
    │   └── dog/
    └── test/
        ├── cat/
        └── dog/

Run the training script:

bash

python main.py

Follow the on-screen prompts to proceed with training and saving the model.