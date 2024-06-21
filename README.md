

# Accident Detection using Deep Learning (CNN)

## Overview

This project is an implementation of an accident detection system using Convolutional Neural Networks (CNN). The system aims to detect accidents from images or video frames, leveraging the power of deep learning to identify and classify incidents accurately. Additionally, the project uses Twilio's API to send notifications when an accident is detected.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Dataset](#dataset)
5. [Model Architecture](#model-architecture)
6. [Training the Model](#training-the-model)
7. [Evaluation](#evaluation)
8. [Usage](#usage)
9. [Twilio Integration](#twilio-integration)
10. [Results](#results)
11. [Contributing](#contributing)
12. [License](#license)

## Introduction

Accident detection systems are crucial in improving emergency response times and potentially saving lives. This project uses a CNN to automatically detect accidents from images or video frames. The model is trained on a labeled dataset of accident and non-accident images, allowing it to learn and recognize patterns associated with accidents.

## Features

- **Automatic Detection**: Detect accidents in real-time from video frames or static images.
- **High Accuracy**: Utilizes a CNN for high-accuracy detection.
- **Notification System**: Integrates with Twilio to send SMS notifications when an accident is detected.
- **Scalable**: Can be integrated into various applications, including traffic monitoring systems and vehicle safety mechanisms.

## Installation

### Prerequisites

- Python 3.7+
- TensorFlow 2.0+
- OpenCV
- NumPy
- Matplotlib
- scikit-learn
- Twilio

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/accident-detection.git
   cd accident-detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: `env\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset should consist of labeled images categorized into two classes: `accident` and `non-accident`. You can create your own dataset or use publicly available datasets. Ensure the data is organized in the following structure:

```
data/
    train/
        accident/
        non-accident/
    validation/
        accident/
        non-accident/
```

## Model Architecture

The CNN model used for accident detection is based on a standard architecture, which includes convolutional layers, pooling layers, and fully connected layers. The architecture can be customized based on specific requirements and dataset characteristics.

## Training the Model

To train the model, follow these steps:

1. Ensure your dataset is prepared and organized as described in the [Dataset](#dataset) section.
2. Run the training script:
   ```bash
   python train.py
   ```
3. The script will split the data into training and validation sets, train the CNN model, and save the best model based on validation accuracy.

## Evaluation

To evaluate the model on a test dataset:

1. Ensure you have a test dataset organized similarly to the training dataset.
2. Run the evaluation script:
   ```bash
   python evaluate.py
   ```

## Usage

To use the trained model for accident detection on new images or video frames:

1. Ensure you have a trained model saved.
2. Run the detection script:
   ```bash
   python detect.py --image_path path_to_image
   ```
   or for video:
   ```bash
   python detect.py --video_path path_to_video
   ```

## Twilio Integration

Twilio is used in this project to send SMS notifications when an accident is detected. You will need a Twilio account to use this feature. Follow these steps to set up Twilio integration:

1. **Sign Up for Twilio**: Create an account on the [Twilio website](https://www.twilio.com/).

2. **Get Your Twilio Credentials**: Obtain your Account SID, Auth Token, and phone number from the Twilio Console.

3. **Set Up Environment Variables**: Store your Twilio credentials in environment variables to keep them secure.
    - **Linux/macOS**:
      ```bash
      export TWILIO_ACCOUNT_SID='your_twilio_account_sid'
      export TWILIO_AUTH_TOKEN='your_twilio_auth_token'
      export TWILIO_PHONE_NUMBER='your_twilio_phone_number'
      ```
    - **Windows**:
      ```bash
      set TWILIO_ACCOUNT_SID='your_twilio_account_sid'
      set TWILIO_AUTH_TOKEN='your_twilio_auth_token'
      set TWILIO_PHONE_NUMBER='your_twilio_phone_number'
      ```

    Alternatively, use a `.env` file:
    ```plaintext
    TWILIO_ACCOUNT_SID=your_twilio_account_sid
    TWILIO_AUTH_TOKEN=your_twilio_auth_token
    TWILIO_PHONE_NUMBER=your_twilio_phone_number
    ```

4. **Load Environment Variables in Your Code**:
    ```python
    from dotenv import load_dotenv
    import os
    from twilio.rest import Client

    load_dotenv()

    TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
    TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
    TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    def send_sms_notification(to, message):
        client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=to)
    ```

5. **Send Notifications**: Use the `send_sms_notification` function to send SMS alerts when an accident is detected.
    ```python
    # Example usage
    if accident_detected:
        send_sms_notification('+1234567890', 'Accident detected!')
    ```

## Results

The results of the accident detection model will be displayed, indicating whether an accident is detected in the given image or video frame. Performance metrics such as accuracy, precision, recall, and F1-score will also be provided.


---

For any questions or issues, please open an issue on GitHub or contact the project maintainer.

Happy coding!
