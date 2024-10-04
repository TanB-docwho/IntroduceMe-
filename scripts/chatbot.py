 # Import libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
def interaction():
    print("Hello! I'm IntroduceMe!, your personal chatbot for Tanisha. How can I help you today?")
    
    while True:
        user_input = input("You: ").strip().lower()  # Get user input and standardize it
        
        if user_input in ["exit", "quit", "goodbye", "Goodbye!"]:
            print("Goodbye! Thank you for taking the time to learn more about your prime candidate 😉! Please visit me anytime!")
            break

        # Match user input to the appropriate response function
        elif "name" in user_input:
            print("IntroduceMe!: My name is IntroduceMe! Tanisha is my creator.")
        
        elif "age" in user_input:
            print("IntroduceMe!: Tanisha is 24 years old.")

        elif "skills" in user_input or "skillset" in user_input:
            print("IntroduceMe!: Tanisha's skills include Machine Learning, Programming in Python, Java, SQL, and experience with TensorFlow and PyTorch, among others.")
        
        elif "background" in user_input or "experience" in user_input:
            print("IntroduceMe!: Tanisha has a background in AI data training, customer service, and software testing.")

        elif "years of experience" in user_input:
            print("IntroduceMe!: Tanisha has more than 4 years of professional experience in the field of AI and machine learning.")
        
        elif "company interest" in user_input:
            print("IntroduceMe!: Tanisha seeks a company with good team energy, collaboration, an encouraged learning environment, and unlimited PTO.")

        elif "purpose" in user_input or "function" in user_input:
            print("IntroduceMe!: I was created to introduce Tanisha to hiring managers and recruiters, making her standout in the application process!")

        elif "why is tanisha better than all my other candidates" in user_input:
            print("IntroduceMe!: Tanisha brings a unique blend of technical skills, creativity, and a proven track record in AI projects, making her an exceptional candidate.")

        elif "interests" in user_input or "hobbies" in user_input:
            print("IntroduceMe!: Tanisha loves the color blue, enjoys 'Doctor Who,' and is a fan of the horror genre! She's always up for a thrilling story or a time travel adventure.")

        elif "education" in user_input or "school" in user_input:
            print("IntroduceMe!: Tanisha graduated high school from Eastern High School in Louisville, Kentucky, and plans to pursue a degree in AI and data analysis at the University of Louisville in 2025.")

        elif "projects" in user_input or "work" in user_input:
            print("IntroduceMe!: Tanisha has worked on projects like DeepmingAI-Spark, Komorebi, Cohere Coral, and Character AI-Cerberus.")

        elif "career goals" in user_input or "future plans" in user_input:
            print("IntroduceMe!: Tanisha aims to create models that will benefit the future of education and gaming for the general public. She believes in harnessing AI for positive societal impact.")

         elif "methodologies" in user_input:
            print("IntroduceMe!: Tanisha prefers methodologies such as supervised learning, reinforcement learning, and continuous integration when working on machine learning projects.")

        elif "how does she stay updated" in user_input:
            print("IntroduceMe!: To stay updated with the latest AI trends and technologies, Tanisha participates in online courses, webinars, and follows industry leaders on social media.")

        elif "challenges she faced" in user_input:
            print("IntroduceMe!: Tanisha has faced challenges like data quality issues and adapting to changing project requirements. She overcomes these through collaboration and continuous learning.")

        elif "teamwork" in user_input:
            print("IntroduceMe!: Tanisha values teamwork and encourages open communication, believing that diverse perspectives lead to better solutions.")

        elif "specialization" in user_input:
            print("IntroduceMe!: Tanisha is particularly excited about natural language processing and reinforcement learning, and she aims to specialize further in these areas.")

        elif "feedback" in user_input:
            print("IntroduceMe!: Tanisha welcomes feedback as an essential part of her growth, using it to improve her skills and refine her work.")
        
            # Add follow-up questions based on existing responses
        elif "tell me more about her projects" in user_input:
            print("IntroduceMe!: Sure! When working with DeepmindAI, Tanisha worked on supervised fine-tuning techniques. With Komorebi, she focused on RAG orchestrations- implementing real-world functions for users to inquiry about their banks accounts, retail shopping, healthcare knowledge, and food plans.

        elif "what's her favorite color" in user_input:
            print("IntroduceMe!: Tanisha's favorite color is blue! It's as vibrant and energetic as her personality.")

        elif "what does she like about AI" in user_input:
            print("IntroduceMe!: Tanisha loves the hands-on nature of AI work because it allows her to see immediate outputs from her inputs, making it a thrilling experience!")

        # Add more elif statements for any other questions you want to handle
        else:
            print("IntroduceMe!: I'm sorry, but I don't quite understand that. Can you please rephrase your question?")

        knowledge_base = {
    "professional_summary": "Highly motivated and detail-oriented professional seeking a Machine Learning Engineer position. Offering a background in client satisfaction, model evaluation, and innovation.",
    
    "work_experience": [
        {
            "job_title": "Advanced AI Data Trainer",
            "company": "Invisible Technologies",
            "dates": "06/2023 - Current",
            "responsibilities": [
                "Enhanced model accuracy by 56% by compiling, cleaning, and manipulating large datasets to ensure precision in training and data integrity.",
                "Improved model reliability and compliance by utilizing machine learning techniques and tagging behaviors based on client safety regulations to produce responsible AI.",
                "Stayed up-to-date with the latest advancements in machine learning and actively participated in knowledge-sharing activities.",
                "Developed and tested prompts for 25 model capabilities, ensuring they met the evolving requirements of RLHF (Reinforcement Learning from Human Feedback) and NLP methodologies, contributing to higher model performance.",
                "Consistently met and exceeded client satisfaction goals (85-100% accuracy) by collaborating with cross-functional teams and adapting swiftly to project updates and client feedback.",
                "Streamlined data testing processes by integrating automation using Python, SQL, and Java, reducing the occurrence of manual errors in model testing."
            ]
        },
        {
            "job_title": "Prompt Engineer",
            "company": "Remotetasks",
            "dates": "03/2020 - 06/2023",
            "responsibilities": [
                "Increased the accuracy of AI model predictions by reformulating and validating models, focusing on continuous testing and quality assurance processes.",
                "Analyzed chatbot responses and refined team prompts, leading to safer and more coherent AI outputs, improving the overall quality of interactions.",
                "Collaborated with cross-functional teams to integrate machine learning solutions into production systems."
            ]
        },
        {
            "job_title": "Software Tester",
            "company": "Pegatron Inc.",
            "dates": "09/2022 - 04/2023",
            "responsibilities": [
                "Increased software reliability by developing and executing test plans for software and hardware products, ensuring they met quality and performance standards.",
                "Identified and addressed over 20 critical defects a week through close collaboration with developers and A/B Testing, leading to smoother product releases and a reduction in post-release issues.",
                "Assisted in collecting, preprocessing, and analyzing data to uncover patterns and insights.",
                "Improved the efficiency of testing procedures by implementing functional compatibility and regression tests, contributing to faster and more accurate product evaluations.",
                "Ensured compliance with industry standards by conducting usability tests and collaborating with cross-functional teams to address design and functionality issues before launch."
            ]
        }
    ],
    "skills": {
    "Machine Learning": [
        "Reinforcement Learning (RLHF)", 
        "Natural Language Processing (NLP)", 
        "Supervised Fine-Tuning (SFT)", 
        "Anomaly Detection", 
        "Feature Engineering", 
        "Neural Networks"
    ],
    "Programming Languages": [
        "Python", 
        "Java", 
        "SQL", 
        "JSON"
    ],
    "ML Frameworks": [
        "TensorFlow", 
        "PyTorch"
    ],
    "Data Engineering": [
        "Data Cleaning", 
        "Data Preprocessing", 
        "Spark", 
        "Large-Scale Data Management", 
        "Data Analysis", 
        "Data Visualization"
    ],
    "Model Development": [
        "A/B Testing", 
        "Cross-Validation", 
        "Model Evaluation"
    ],
    "Risk and Trust Domains": [
        "Safety Annotation", 
        "Trust & Risk Mitigation"
    ],
    "Collaboration Tools": [
        "Slack", 
        "JIRA", 
        "Confluence"
    ],
    "Other": [
        "Technical Writing", 
        "Prompt Engineering", 
        "Test-Driven Development", 
        "Strong Problem Solving", 
        "Attention to Detail", 
        "Teamwork", 
        "Excellent Communication", 
        "Comfortable Reading Research Papers"
    ]
}


    "projects": [
        "DeepmingAI-Spark (SFT)",
        "Komorebi (RAG Orchestrations)",
        "Cohere Coral (RLHF) (SFT)",
        "Character AI-Cerberus (Safety Annotation)"
    ],
    
    "company_preferences": {
        "wants": [
            "Good team energy",
            "Collaboration",
            "Encouraged learning environment",
            "Unlimited PTO",
            "Challenging projects"
        ]
    },
    "career_goals": (
            "Tanisha aims to create models that will benefit the future of education and gaming for the general public. "
            "She believes in harnessing AI for positive societal impact."
        ),
        "methodologies": (
            "Tanisha prefers methodologies such as supervised learning, reinforcement learning, and continuous integration when working on machine learning projects."
        ),
        "keeping_updated": (
            "To stay updated with the latest AI trends and technologies, Tanisha regularly participates in online courses, webinars, and industry conferences. "
            "She also follows prominent AI researchers and practitioners on social media."
        ),
        "challenges": (
            "Tanisha faced several challenges in her roles, such as dealing with data quality issues and adapting to rapidly changing project requirements. "
            "She overcame these challenges through collaboration, continuous learning, and by leveraging her problem-solving skills."
        ),
        "teamwork": (
            "Tanisha values teamwork and collaboration, believing that diverse perspectives lead to better solutions. "
            "She encourages open communication and actively seeks input from her teammates."
        ),
        "specialization": (
            "Tanisha finds the areas of natural language processing and reinforcement learning particularly exciting and aims to specialize further in these fields."
        ),
        "feedback_handling": (
            "If you have any feedback for Tanisha about her resume, linkden profile, or me!, please let me know and i will make sure it gets back to her :)"
        )
    }
}

         
    "creation_origin": (
        "Tanisha created me as her first chatbot project to utilize and tinker with her machine learning and AI building skills. "
        "In total, it took her about 13 hours and 45 minutes. She wanted to stand out from the competition and have a way for hiring managers "
        "and recruiters to get to know her without going through hundreds of long applications. If for nothing else then for fun! Just to see what she could do ☺️"
    ),
    
    "personal_information": {
        "name": "Tanisha Boston",
        "age": 24,
        "industry_interest": [
            "Tanisha’s mom was a technical writer, and it was the first time she ever saw her code.",
            "Started training AI on remote tasks in 2020 and fell in love with the process ever since.",
            "She loves hands-on work, and this career allows her to see immediate output from her input."
        ],
        "education": {
            "high_school": {
                "name": "Eastern High School",
                "location": "Louisville, Kentucky",
                "graduation_year": 2018,
                "achievements": "Recognized in Japanese language and culture."
            },
            "college": {
                "intended_institution": "University of Louisville",
                "degree": "AI and Data Analysis",
                "major_focus": "Machine Learning",
                "start_year": 2025,
                "predicted_graduation": 2028
            }
        },
        "likes": [
            "The color blue",
            "Doctor Who (loves time travel, Syfy stuff)",
            "Anything from the Horror genre"
        ]
    }
}

   
