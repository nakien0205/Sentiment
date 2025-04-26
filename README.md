# Sentiment Analysis Chatbot

A web-based chatbot that can detect emotions in text and respond accordingly. This chatbot uses a pre-trained Hugging Face model to analyze sentiment and generate appropriate responses based on detected emotions.

## Features

- **Emotion Detection**: Detects 7 emotions (happiness, sadness, fear, anger, surprise, disgust, and neutral)
- **Contextual Responses**: Provides responses tailored to the detected emotion
- **Dual Interface**: Supports both text and voice interaction
- **Conversation History**: Maintains chat history throughout the session
- **Responsive Design**: Works on both desktop and mobile devices

## Technical Details

### Components

1. **Sentiment Analysis Model**: Uses the `j-hartmann/emotion-english-distilroberta-base` model from Hugging Face
2. **Response Generator**: Custom logic for generating appropriate responses based on detected emotions
3. **Web Interface**: Flask-based web application with both text and voice capabilities
4. **Speech Recognition**: Browser-based speech recognition for voice input
5. **Text-to-Speech**: Browser-based speech synthesis for voice output

### Dependencies

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- Flask
- SpeechRecognition
- pyttsx3
- NumPy

## Installation

1. Clone the repository or download the source code
2. Install the required dependencies:
   ```
   pip install torch transformers flask SpeechRecognition pyttsx3 numpy
   ```
3. Navigate to the project directory and run the application:
   ```
   python app.py
   ```
4. Access the chatbot in your web browser at `http://localhost:5000`

## Usage Instructions

### Text Chat Mode

1. The default mode is text chat
2. Type your message in the input field at the bottom of the screen
3. Press Enter or click the send button (arrow icon) to send your message
4. The chatbot will analyze your message, detect the emotion, and respond accordingly
5. The detected emotion will be displayed below the chatbot's response

### Voice Chat Mode

1. Click the "Voice Chat" button in the mode toggle at the top of the chat interface
2. Click the microphone button to start recording
3. Speak your message clearly
4. The recording will automatically stop when you finish speaking
5. The chatbot will process your speech, detect the emotion, and respond both textually and verbally

### Emotion-Specific Responses

The chatbot responds differently based on the detected emotion:

- **Sadness**: Identifies what makes you sad and asks if you want advice or just to be heard
- **Fear**: Identifies your fear and provides encouragement or advice
- **Anger**: Identifies what makes you angry and attempts to calm you with reasoning
- **Happiness**: Responds with shared enthusiasm and asks what's bringing you joy
- **Surprise**: Acknowledges the surprising nature and encourages you to share more
- **Disgust**: Validates your feeling of disgust and offers to change the subject
- **Neutral**: Provides a balanced response and asks if there's anything specific you'd like to discuss

## Customization

### Adding New Emotions

To add new emotions to the detection model, you would need to:
1. Fine-tune the model on a dataset that includes the new emotions
2. Update the `emotion_labels` dictionary in the `SentimentAnalyzer` class
3. Add corresponding response methods in the `ResponseGenerator` class

### Modifying Response Strategies

To change how the chatbot responds to specific emotions:
1. Edit the corresponding response method in the `ResponseGenerator` class
2. For example, to modify how the chatbot responds to sadness, edit the `_respond_to_sadness` method

## Troubleshooting

- **Voice Recognition Issues**: Make sure your browser supports the Web Speech API (Chrome, Edge, and Safari are recommended)
- **Model Loading Errors**: Ensure you have a stable internet connection when first running the application, as it needs to download the model
- **Response Latency**: The first few responses might be slower as the model loads and warms up

## Future Improvements

- Add multi-language support
- Implement more sophisticated conversation context tracking
- Add user authentication and personalized response patterns
- Integrate with external APIs for more dynamic responses
- Improve voice recognition accuracy with custom models

## License

This project is open source and available under the MIT License.
