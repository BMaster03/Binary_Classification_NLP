from src.Binary_C_NLP import BinaryClassification

def main():
    """
        Main function to execute the binary classification workflow on the IMDB dataset.

        This function initializes the BinaryClassification class, prepares the data,
        trains the model, plots the training history, and evaluates the model on the test data.
    """

    binary_classifier = BinaryClassification() # Initialize the BinaryClassification class
    binary_classifier.run() # Execute the binary classification workflow

if __name__ == "__main__":
    main()