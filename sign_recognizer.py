import numpy as np


class SignRecognizer:
    """
    Handles sign language recognition
    """
    def __init__(self, model_path=None):
        # Label mapping
        self.labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        # TODO: use learning model
        # Load model
        self.model = None


    def predict(self, roi):
        """
        Predict sign language letter from ROI
        Returns: (predicted_char, confidence)
        """
        if roi is None:
            return None, 0.0

        if self.model is None:
            # TESTING AREA
            return "A", np.random.uniform(0.5, 1.0) #np.random.choice(self.labels), np.random.uniform(0.5, 1.0)

        # Prepare input for model
        #scale
        #input_data = np.expand_dims(roi, axis=0)

        # Get prediction
        prediction = self.model.predict(roi)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        return self.labels[predicted_class], confidence