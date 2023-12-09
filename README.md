# NameSimFinder

## Overview
NameSimFinder is a Django REST API designed for a knowledge management system. It specializes in finding the sentences that are most similar to a given sentence within a CSV file. This project utilizes MongoDB for storing embedded sentences and employs ParsBERT for processing Farsi sentences. It converts sentences into vectors and identifies similar sentences using cosine similarities.

## Installation
To set up NameSimFinder, clone the repository and install the required libraries from the `requirements.txt` file using pip:
```
pip install -r requirements.txt
```

## Usage
Start the Django server to use the API. It provides two main endpoints:

1. **/api/findSims/**: Finds and returns similar sentences. Sample JSON body for the POST request:
    ```json
    {
        "sentence": "رفع مشکل کیبوردها",
        "numOfSimilars": 10
    }
    ```
    Here, `numOfSimilars` specifies the number of similar sentences to return.

2. **/api/addDoc/**: Adds a new sentence to the database. Sample JSON body for the POST request:
    ```json
    {
        "sentence": "ایران",
        "KID": "7346103"
    }
    ```
    `KID` is an identifier for the sentence.

## Contributing
Contributions are welcome. Please feel free to fork the repository and submit pull requests.
