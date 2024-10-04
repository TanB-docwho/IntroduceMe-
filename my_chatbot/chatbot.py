import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load model and tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large", clean_up_tokenization_spaces=True)
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    model.config.pad_token_id = tokenizer.eos_token_id  # Set pad token id
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Indicate that the model has finished loading
print("Model and tokenizer successfully loaded! Ready for interaction.")

# Load intent classification pipeline
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def generate_response(prompt, temperature=0.7):
    # Tokenize the prompt and send to device (assuming GPU setup)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate response from the model
    with torch.no_grad():  # Disable gradient calculations
        output = model.generate(
            **inputs, 
            max_length=250,  # Adjust as needed
            num_return_sequences=1,
            temperature=temperature,  # Set temperature for response diversity
            do_sample=True,  # Enable sampling
            pad_token_id=tokenizer.eos_token_id  # Avoid generation errors if pad_token is needed
        )

    # Decode the generated output and clean it up
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    # Check if the response starts with the chatbot's name
    if response.startswith("IntroduceMe!: "):
        # Generate a new response if it starts with the chatbot's name
        return generate_response(prompt, temperature)
    
    return response

# Define a basic intent classification system using keyword matching
intent_keywords = {
    "age": ["how old", "age"],
    "education": ["school", "college", "university", "education"],
    "skills": ["skills", "techniques"],
    "ml frameworks": ["ml frameworks",  "frameworks"],
    "work commitment": ["work preferences", "work style", "remote", "hybrid", "onsite"],
    "positions": ["positions", "job titles", "job positions"],
    "company preferences": ["company environment", "company culture", "want in a company", "look for in a company"],
    "career goals": ["career goals", "goals", "aims", "objectives"],
    "methodologies": ["methodologies", "approaches", "techniques"],
    "keeping updated": ["keeping updated", "staying updated", "latest trends"],
    "challenges": ["challenges", "difficulties", "problems"],
    "teamwork": ["teamwork", "collaboration", "working with others"],
    "specialization": ["specialization", "focus areas", "areas of expertise"],
    "feedback handling": ["feedback handling", "feedback", "handling feedback"],
    "creation origin": ["creation origin", "how you were created", "origin story", "purpose"],
    "personal information": ["how did she get her start", "initial interest", "how did she get into this industry"],
    "likes": ["likes", "favorite", "fun fact", "fun facts"],
    "full time": ["full time", "full-time"],
    "part time": ["part time", "part-time"],
    "remote preference": ["remote work", "work from home"],
    "projects": ["projects", "tasks"],
}

detailed_skill_descriptions = {
    "Machine Learning": (
        "Tanisha has extensive experience with various machine learning techniques, including Reinforcement Learning with Human Feedback (RLHF), which is used to improve model interactions by incorporating user feedback. Her expertise in Natural Language Processing (NLP) enables her to design and deploy models that understand and generate human language effectively. She’s also skilled in prompt engineering for optimizing model outputs and anomaly detection for identifying patterns that don’t conform to expected behaviors. With experience in Supervised Fine-Tuning (SFT) and Conversational AI, Tanisha has contributed to building interactive AI systems, leveraging neural networks for deep learning tasks."
    ),
    "Software Engineering": (
        "In software engineering, Tanisha applies Object-Oriented Programming (OOP) principles to build scalable, maintainable codebases. She has a solid understanding of data structures and algorithms, essential for solving complex computational problems. She uses version control (Git) to track code changes and collaborate with other developers, ensuring code integrity. Her knowledge of Node.js and React allows her to build web-based applications, while her attention to code reviews ensures that her projects meet high-quality standards."
    ),
    "Programming Languages": (
        "Tanisha is highly proficient in Python, the language she uses to build and deploy machine learning models. She is familiar with Java for object-oriented programming and has worked with SQL for database queries and management. She also frequently uses JSON for data interchange between systems."
    ),
    "ML Frameworks": (
        "Tanisha is skilled in using machine learning frameworks like TensorFlow and PyTorch to develop, train, and optimize machine learning models. She leverages NLTK for natural language processing tasks and works within Jupyter notebooks to write, test, and document her code."
    ),
    "Responsible AI": (
        "Tanisha is committed to Responsible AI, focusing on bias mitigation and ensuring her models are fair and unbiased. She applies explainability techniques to make AI decision-making more transparent, using methods like SHAP and LIME. She integrates Human-in-the-Loop (HITL) strategies to refine models through human feedback and adheres to ethical AI development frameworks. Her work in adversarial robustness helps protect AI systems from manipulation, while her emphasis on algorithmic accountability and transparency ensures that her models are both auditable and understandable."
    ),
    "Data Engineering": (
        "Tanisha has hands-on experience in data engineering, from data cleaning and preprocessing to handling large-scale data management. She excels in data analysis to draw actionable insights and leverages data visualization tools to present her findings in an intuitive and impactful way."
    ),
    "Model Development": (
        "Tanisha is proficient in model development techniques, including A/B testing to compare model performance and cross-validation to assess model accuracy. Her ability to perform thorough model evaluation ensures that her models meet performance and reliability standards before deployment."
    ),
    "Risk and Trust Domains": (
        "Tanisha focuses on safety annotation and trust & risk mitigation in AI systems, ensuring that models are safe, fair, and transparent. She implements risk assessment techniques to minimize potential harm, particularly in high-stakes domains, and works to build trust in the AI systems she develops."
    ),
    "Collaboration Tools": (
        "Tanisha regularly uses Slack, JIRA, and Confluence to collaborate with teams across projects. These tools help her manage workflows, track tasks, and maintain seamless communication throughout the project lifecycle."
    ),
    "Other Skills": (
        "Tanisha’s technical writing skills ensure clear documentation and effective communication of complex machine learning concepts. She practices Test-Driven Development (TDD) to write robust, bug-free code. Her strengths in problem-solving, attention to detail, and teamwork make her a key contributor to any project, while her communication skills allow her to engage with both technical and non-technical stakeholders. Additionally, her ability to perform research paper analysis keeps her at the forefront of the latest advancements in AI."
    )
}
def classify_intent(user_input, context):
    user_input_lower = user_input.lower()
    for intent, keywords in intent_keywords.items():
        if any(keyword in user_input_lower for keyword in keywords):
            return intent
    
    # Use zero-shot classification for better intent detection
    candidate_labels = list(intent_keywords.keys())
    result = intent_classifier(" ".join(context), candidate_labels)
    return result['labels'][0]

def extract_name(user_input):
    # Define regex patterns to capture names
    patterns = [
        r"I'm\s+([\w\s]+)",        # Matches "I'm [name]"
        r"I am\s+([\w\s]+)",       # Matches "I am [name]"
        r"My name is\s+([\w\s]+)", # Matches "My name is [name]"
        r"Name's\s+([\w\s]+)",     # Matches "Name's [name]"
        r"Call me\s+([\w\s]+)",    # Matches "Call me [name]"
        r"Hi, I'm\s+([\w\s]+)",    # Matches "Hi, I'm [name]"
        r"Hi, I am\s+([\w\s]+)",   # Matches "Hi, I am [name]"
        r"Hi, call me\s+([\w\s]+)" # Matches "Hi, call me [name]"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            return match.group(1).strip().capitalize()  # Capitalizes the name directly

    return "Guest"  # Default if no name is found

def store_feedback(feedback):
    with open("feedback.txt", "a") as file:
        file.write(feedback + "\n")
    print("Feedback stored. Thank you!")

def slow_print(text, delay=0.03):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()  # Print a newline at the end

def interaction():
    print("Hello! I'm IntroduceMe!, your personal guide for getting to know Tanisha. What's your name?")
    user_name = input("User: ")
    user_name = extract_name(user_name)
    slow_print(f"IntroduceMe!: Hello, {user_name}! What would you like to know first?")
    
    context = []
    feedback = []
    last_input_time = time.time()
    prompt_shown = False
    
    while True:
        user_input = input(f"{user_name}: ")
        
        # Exit condition
        if user_input.lower() in ["exit", "quit", "bye", "goodbye", "that's all", "no more questions"]:
            slow_print("IntroduceMe!: Goodbye! Come back anytime to learn more about your prime candidate ;)!")
            break
        
        # Update context
        context.append(user_input)
        
        # Classify intent
        intent = classify_intent(user_input, context)
        
        if intent == "age":
            generated_response = "Tanisha is 24 years old."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "education":
            generated_response = "Tanisha attented a technology focused highschool, Eastern High, from 2014 to 2018. Now, she is pursuing a Computer Science & Engineering degree with a focus on machine learning at University of Louisville. She estimates to graduate in 2028."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "skills":
            generated_response = "Tanisha has a wide range of skills in machine learning, software engineering, programming languages, ML frameworks, responsible AI, data engineering, model development, risk and trust domains, collaboration tools, and other skills."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "ml frameworks":
            generated_response = "Tanisha has experience with ML frameworks like TensorFlow, PyTorch, Jupyter and Scikit-Learn."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "work commitment":
            generated_response = "Tanisha prefers fully remote work. While she is open to traveling for collaboration a few times a year, she is not looking for hybrid or on-site positions."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "positions":
            generated_response = "Tanisha is seeking a Machine Learning Engineer or AI Engineer."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "company preferences":
            generated_response = "Tanisha looks for Good team energy, Collaboration, Encouraged learning environment, Unlimited PTO, and Challenging projects in a company."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "career goals":
            generated_response = "Tanisha aims to create models that will benefit the future of education and gaming for the general public. She believes in harnessing AI for positive societal impact."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "methodologies":
            generated_response = "Tanisha prefers methodologies such as supervised learning, reinforcement learning, and continuous integration when working on machine learning projects."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "keeping updated":
            generated_response = "To stay updated with the latest AI trends and technologies, Tanisha regularly participates in online courses, webinars, and industry conferences. She also follows prominent AI researchers and practitioners on social media."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "challenges":
            generated_response = "Tanisha faced several challenges in her roles, such as dealing with data quality issues and adapting to rapidly changing project requirements. She overcame these challenges through collaboration, continuous learning, and by leveraging her problem-solving skills."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "teamwork":
            generated_response = "Tanisha values teamwork and collaboration, believing that diverse perspectives lead to better solutions. She encourages open communication and actively seeks input from her teammates."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "specialization":
            generated_response = "Tanisha finds the areas of natural language processing and reinforcement learning particularly exciting and aims to specialize further in these fields."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "feedback handling":
            generated_response = "If you have any feedback for Tanisha about her resume, LinkedIn profile, or me!, please let me know and I will make sure it gets back to her :)"
            slow_print(f"IntroduceMe!: {generated_response}")
            feedback.append(user_input)
            store_feedback(user_input)
        
        elif intent == "creation origin":
            generated_response = "Tanisha created me as her first chatbot project to utilize and tinker with her machine learning and AI building skills. In total, it took her about 6 days. She wanted to stand out from the competition and have a way for hiring managers and recruiters to get to know her better without going through hundreds of long applications!"
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "personal information":
            generated_response = "Tanisha’s mom was a technical writer, and it was the first time she ever saw her code. Started training AI on remote tasks in 2020 and fell in love with the process ever since. She loves hands-on work, and this career allows her to see immediate output from her input."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "likes":
            generated_response = "Tanisha likes the color blue, Doctor Who (loves time travel, Syfy stuff), and anything from the Horror genre."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "full time":
            generated_response = "Yes, Tanisha is looking for full-time positions."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "part time":
            generated_response = "No, she is not seeking part-time work at this time."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "remote preference":
            generated_response = "Tanisha prefers fully remote work. While she is open to traveling for collaboration a few times a year, she is not looking for hybrid or on-site positions."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent == "projects":
            generated_response = "Tanisha has worked on several projects, including a sentiment analysis tool for social media, a recommendation system for e-commerce, and a predictive model for customer churn. She has also contributed to open-source projects in the machine learning community."
            slow_print(f"IntroduceMe!: {generated_response}")
        
        elif intent in detailed_skill_descriptions:
            # Handle specific skill queries with detailed descriptions
            generated_response = detailed_skill_descriptions[intent]
            slow_print(f"IntroduceMe!: {generated_response}")
        
        else:
            # Fallback response for unrecognized intents
            generated_response = generate_response(" ".join(context))
            slow_print(f"IntroduceMe!: {generated_response}")
        
        # Check if user takes more than 10 seconds to ask something or if they specifically ask for more questions
        current_time = time.time()
        if not prompt_shown and (current_time - last_input_time > 60 or "anything else" in user_input.lower()):
            slow_print("IntroduceMe!: Would you like to learn more about Tanisha's background, academic career, or skills?")
            prompt_shown = True
        
        last_input_time = current_time
    
    # Prompt for feedback near the end of the conversation
    feedback_input = input("IntroduceMe!: Do you have any feedback for Tanisha about her resume, LinkedIn profile, or me? (yes/no): ")
    if feedback_input.lower() in ["yes", "y"]:
        feedback_text = input("IntroduceMe!: Please provide your feedback: ")
        store_feedback(feedback_text)
    else:
        slow_print("IntroduceMe!: Thank you for your time! If you think of any feedback later, feel free to share it.")

# Start the interaction loop
interaction()








        

   