"""
T5 Model Integration with LoRA for Emotion-Aware Chatbot

This script implements the T5 model with LoRA fine-tuning for generating
diverse, emotion-aware responses in a chatbot.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import os

class T5EmotionChatbot:
    """
    A chatbot that uses T5 with LoRA fine-tuning to generate emotion-aware responses.
    """
    
    def __init__(self, model_name="google/flan-t5-base", lora_r=8, lora_alpha=32, lora_dropout=0.1):
        """
        Initialize the T5 emotion-aware chatbot.
        
        Args:
            model_name (str): The name of the base T5 model to use
            lora_r (int): LoRA rank parameter
            lora_alpha (int): LoRA alpha parameter
            lora_dropout (float): LoRA dropout rate
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Define LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"],  # Attention query and value matrices
        )
        
        # Apply LoRA to the model
        self.peft_model = get_peft_model(self.model, self.lora_config)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Define emotion categories
        self.emotions = [
            "happiness", "sadness", "anger", "fear", 
            "surprise", "disgust", "neutral", "excitement",
            "gratitude", "love", "confusion", "disappointment"
        ]
    
    def prepare_datasets(self):
        """
        Prepare and process the datasets for fine-tuning.
        
        Returns:
            dict: Processed datasets for training and evaluation
        """
        # Load DailyDialog dataset
        daily_dialog = load_dataset("daily_dialog")
        
        # Load GoEmotions dataset
        go_emotions = load_dataset("go_emotions")
        
        # Process DailyDialog for conversational context
        def process_daily_dialog(examples):
            processed_examples = {"input": [], "output": []}
            
            for dialog, emotion in zip(examples["dialog"], examples["emotion"]):
                for i in range(len(dialog) - 1):
                    # Get emotion of current utterance
                    emotion_idx = emotion[i]
                    emotion_label = self._map_daily_dialog_emotion(emotion_idx)
                    
                    # Create input with emotion context
                    input_text = f"Emotion: {emotion_label} Context: {dialog[i]}"
                    output_text = dialog[i + 1]
                    
                    processed_examples["input"].append(input_text)
                    processed_examples["output"].append(output_text)
            
            return processed_examples
        
        # Process GoEmotions for emotion understanding
        def process_go_emotions(examples):
            processed_examples = {"input": [], "output": []}
            
            for text, labels in zip(examples["text"], examples["labels"]):
                if len(labels) > 0:
                    # Get the first emotion label
                    emotion_idx = labels[0]
                    emotion_label = self._map_go_emotions_label(emotion_idx)
                    
                    # Create input with emotion detection task
                    input_text = f"Detect emotion: {text}"
                    output_text = f"Emotion: {emotion_label}"
                    
                    processed_examples["input"].append(input_text)
                    processed_examples["output"].append(output_text)
            
            return processed_examples
        
        # Apply processing functions
        daily_dialog_processed = daily_dialog.map(
            process_daily_dialog, 
            batched=True, 
            remove_columns=daily_dialog["train"].column_names
        )
        
        go_emotions_processed = go_emotions.map(
            process_go_emotions, 
            batched=True, 
            remove_columns=go_emotions["train"].column_names
        )
        
        # Combine datasets
        combined_train = {
            "input": daily_dialog_processed["train"]["input"] + go_emotions_processed["train"]["input"],
            "output": daily_dialog_processed["train"]["output"] + go_emotions_processed["train"]["output"]
        }
        
        combined_validation = {
            "input": daily_dialog_processed["validation"]["input"] + go_emotions_processed["validation"]["input"],
            "output": daily_dialog_processed["validation"]["output"] + go_emotions_processed["validation"]["output"]
        }
        
        # Tokenize datasets
        def tokenize_function(examples):
            model_inputs = self.tokenizer(
                examples["input"],
                max_length=512,
                padding="max_length",
                truncation=True
            )
            
            labels = self.tokenizer(
                examples["output"],
                max_length=128,
                padding="max_length",
                truncation=True
            ).input_ids
            
            model_inputs["labels"] = labels
            return model_inputs
        
        tokenized_train = tokenize_function(combined_train)
        tokenized_validation = tokenize_function(combined_validation)
        
        return {
            "train": tokenized_train,
            "validation": tokenized_validation
        }
    
    def fine_tune(self, output_dir="./t5_emotion_chatbot", num_epochs=3, batch_size=8, learning_rate=1e-4):
        """
        Fine-tune the T5 model with LoRA on the prepared datasets.
        
        Args:
            output_dir (str): Directory to save the fine-tuned model
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for training
        """
        from transformers import Trainer, TrainingArguments
        
        # Prepare datasets
        datasets = self.prepare_datasets()
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            learning_rate=learning_rate,
        )
        
        # Create Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
        )
        
        # Start training
        trainer.train()
        
        # Save the fine-tuned model
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load_fine_tuned_model(self, model_path):
        """
        Load a fine-tuned model from the specified path.
        
        Args:
            model_path (str): Path to the fine-tuned model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.peft_model = PeftModel.from_pretrained(self.model, model_path)
    
    def detect_emotion(self, text):
        """
        Detect the emotion in the given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            str: The detected emotion
        """
        input_text = f"Detect emotion: {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=20,
                num_return_sequences=1
            )
        
        emotion_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract emotion from the output
        if "Emotion:" in emotion_text:
            emotion = emotion_text.split("Emotion:")[1].strip().lower()
            if emotion in self.emotions:
                return emotion
        
        return "neutral"  # Default to neutral if no valid emotion is detected
    
    def generate_response(self, user_input, max_history=5):
        """
        Generate a response based on the user input and conversation history.
        
        Args:
            user_input (str): The user's message
            max_history (int): Maximum number of turns to include in history
            
        Returns:
            str: The generated response
        """
        # Detect emotion in user input
        emotion = self.detect_emotion(user_input)
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare context from conversation history
        history_text = ""
        if len(self.conversation_history) > 1:
            # Get the last few turns (limited by max_history)
            recent_history = self.conversation_history[-max_history*2:-1] if len(self.conversation_history) > max_history*2 else self.conversation_history[:-1]
            
            for entry in recent_history:
                role = "User" if entry["role"] == "user" else "Assistant"
                history_text += f"{role}: {entry['content']}\n"
        
        # Create input with emotion and history context
        input_text = f"Emotion: {emotion} Context: {history_text}User: {user_input}\nAssistant:"
        
        # Tokenize input
        inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate response
        with torch.no_grad():
            outputs = self.peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_return_sequences=1,
                temperature=0.8,  # Add some randomness for diversity
                top_p=0.9,  # Use nucleus sampling for more natural responses
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Limit conversation history size
        if len(self.conversation_history) > max_history * 4:
            self.conversation_history = self.conversation_history[-max_history*2:]
        
        return response
    
    def _map_daily_dialog_emotion(self, emotion_idx):
        """Map DailyDialog emotion indices to emotion labels."""
        emotion_map = {
            0: "neutral",
            1: "happiness",
            2: "sadness",
            3: "anger",
            4: "surprise",
            5: "fear",
            6: "disgust"
        }
        return emotion_map.get(emotion_idx, "neutral")
    
    def _map_go_emotions_label(self, emotion_idx):
        """Map GoEmotions indices to emotion labels."""
        # This is a simplified mapping - GoEmotions has 27 emotions
        # We map them to our simplified set of emotions
        go_emotions_map = {
            0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
            4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
            8: "desire", 9: "disappointment", 10: "disapproval", 11: "disgust",
            12: "embarrassment", 13: "excitement", 14: "fear", 15: "gratitude",
            16: "grief", 17: "joy", 18: "love", 19: "nervousness",
            20: "optimism", 21: "pride", 22: "realization", 23: "relief",
            24: "remorse", 25: "sadness", 26: "surprise", 27: "neutral"
        }
        
        # Map to our simplified emotion set
        emotion_name = go_emotions_map.get(emotion_idx, "neutral")
        
        # Map to our simplified emotion categories
        simplified_map = {
            "admiration": "happiness", "amusement": "happiness", "anger": "anger",
            "annoyance": "anger", "approval": "happiness", "caring": "love",
            "confusion": "confusion", "curiosity": "surprise", "desire": "love",
            "disappointment": "disappointment", "disapproval": "disgust", "disgust": "disgust",
            "embarrassment": "sadness", "excitement": "excitement", "fear": "fear",
            "gratitude": "gratitude", "grief": "sadness", "joy": "happiness",
            "love": "love", "nervousness": "fear", "optimism": "happiness",
            "pride": "happiness", "realization": "surprise", "relief": "happiness",
            "remorse": "sadness", "sadness": "sadness", "surprise": "surprise",
            "neutral": "neutral"
        }
        
        return simplified_map.get(emotion_name, "neutral")

# Example usage
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = T5EmotionChatbot()
    
    # Example of preparing datasets and fine-tuning
    # Note: This would require significant compute resources
    # chatbot.fine_tune()
    
    # Example of loading a fine-tuned model
    # chatbot.load_fine_tuned_model("./t5_emotion_chatbot")
    
    # Example of generating a response
    # response = chatbot.generate_response("I'm feeling really happy today!")
    # print(response)
