# IntroduceMe! Chatbot

## Overview

IntroduceMe! is a sophisticated chatbot designed to help users get to know Tanisha, a candidate with extensive experience in machine learning and software engineering. The chatbot leverages advanced NLP techniques and machine learning models to provide interactive and informative responses. It is built using PyTorch, Hugging Face's Transformers library, and Flask for web integration.

## Features

### Advanced NLP and Machine Learning
- **Model Integration**: Utilizes the `google/flan-t5-large` model for generating human-like responses.
- **Intent Classification**: Implements a zero-shot classification pipeline using `facebook/bart-large-mnli` for accurate intent detection.
- **Detailed Skill Descriptions**: Provides in-depth descriptions of Tanisha's skills in various domains, including Machine Learning, Software Engineering, and Responsible AI.

### User Interaction
- **Name Extraction**: Automatically extracts and capitalizes the user's name from their input.
- **Contextual Understanding**: Maintains a conversation context to provide coherent and relevant responses.
- **Feedback Mechanism**: Allows users to provide feedback on Tanisha's resume, LinkedIn profile, or the chatbot itself.

### Technical Details
- **GPU Acceleration**: Utilizes GPU for faster response generation if available.
- **Slow Print**: Employs a slow print function to simulate a more natural conversation flow.
- **Exit Conditions**: Gracefully handles exit commands to end the conversation.

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TanB-docwho/IntroduceMe-.git
   cd IntroduceMe-