"""
CBSE Student Response Data Generator
Generates synthetic student question-response data for adaptive learning
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

# CBSE Class 9-10 Biology Topics
TOPICS = {
    'class_9': {
        'cell_biology': {
            'subtopics': ['Cell Structure', 'Cell Organelles', 'Cell Division'],
            'difficulty_range': (1, 5)
        },
        'tissues': {
            'subtopics': ['Plant Tissues', 'Animal Tissues', 'Tissue Functions'],
            'difficulty_range': (1, 4)
        },
        'diversity_organisms': {
            'subtopics': ['Classification', 'Monera', 'Protista', 'Fungi', 'Plantae', 'Animalia'],
            'difficulty_range': (2, 5)
        },
        'health_diseases': {
            'subtopics': ['Infectious Diseases', 'Non-infectious Diseases', 'Prevention'],
            'difficulty_range': (1, 4)
        },
        'natural_resources': {
            'subtopics': ['Soil', 'Water', 'Air', 'Conservation'],
            'difficulty_range': (1, 3)
        }
    },
    'class_10': {
        'life_processes': {
            'subtopics': ['Nutrition', 'Respiration', 'Transportation', 'Excretion'],
            'difficulty_range': (2, 5)
        },
        'reproduction': {
            'subtopics': ['Asexual Reproduction', 'Sexual Reproduction', 'Human Reproduction'],
            'difficulty_range': (2, 5)
        },
        'heredity_evolution': {
            'subtopics': ['Mendel Laws', 'Genetics', 'Evolution', 'Natural Selection'],
            'difficulty_range': (3, 5)
        },
        'environment': {
            'subtopics': ['Ecosystem', 'Food Chain', 'Environmental Issues', 'Biodiversity'],
            'difficulty_range': (1, 4)
        },
        'sustainable_development': {
            'subtopics': ['Natural Resources', 'Conservation', 'Sustainable Practices'],
            'difficulty_range': (1, 3)
        }
    }
}

def generate_question_bank():
    """Generate a question bank with questions of varying difficulty"""
    questions = []
    question_id = 1

    for class_level, topics in TOPICS.items():
        for topic, info in topics.items():
            min_diff, max_diff = info['difficulty_range']

            for subtopic in info['subtopics']:
                # Generate 5-10 questions per subtopic
                n_questions = np.random.randint(5, 11)

                for _ in range(n_questions):
                    difficulty = np.random.randint(min_diff, max_diff + 1)

                    # Question types
                    q_type = np.random.choice(['MCQ', 'True/False', 'Short Answer'], p=[0.6, 0.2, 0.2])

                    # Estimated time (seconds) based on difficulty
                    est_time = 30 + difficulty * 20 + np.random.randint(-10, 20)

                    questions.append({
                        'question_id': f'Q{question_id:04d}',
                        'class_level': class_level,
                        'topic': topic,
                        'subtopic': subtopic,
                        'difficulty': difficulty,
                        'question_type': q_type,
                        'estimated_time_sec': est_time,
                        'max_marks': 1 if q_type == 'MCQ' else (1 if q_type == 'True/False' else 2)
                    })
                    question_id += 1

    return pd.DataFrame(questions)

def generate_student_profiles(n_students=200):
    """Generate student profiles with varying ability levels"""
    students = []

    for i in range(n_students):
        # Student ability follows normal distribution
        ability = np.clip(np.random.normal(0.6, 0.15), 0.2, 0.95)

        # Some topics are stronger than others
        topic_abilities = {}
        for class_level, topics in TOPICS.items():
            for topic in topics.keys():
                # Topic ability varies around overall ability
                topic_abilities[topic] = np.clip(ability + np.random.normal(0, 0.1), 0.1, 0.95)

        students.append({
            'student_id': f'STU{i+1:04d}',
            'name': f'Student_{i+1}',
            'class_level': np.random.choice(['class_9', 'class_10']),
            'overall_ability': round(ability, 3),
            'topic_abilities': topic_abilities,
            'learning_speed': np.random.choice(['slow', 'medium', 'fast'], p=[0.2, 0.6, 0.2]),
            'practice_frequency': np.random.choice(['daily', 'weekly', 'irregular'], p=[0.3, 0.5, 0.2])
        })

    return students

def simulate_response(student, question):
    """Simulate a student's response to a question using IRT model"""
    # Get student's ability for this topic
    topic_ability = student['topic_abilities'].get(question['topic'], student['overall_ability'])

    # IRT 2-parameter model: P(correct) = 1 / (1 + exp(-a*(ability - difficulty)))
    # a = discrimination parameter (fixed at 1.5 for simplicity)
    a = 1.5
    difficulty_scaled = (question['difficulty'] - 3) / 2  # Scale difficulty to roughly -1 to 1

    # Probability of correct answer
    logit = a * (topic_ability - 0.5 - difficulty_scaled * 0.3)
    prob_correct = 1 / (1 + np.exp(-logit))

    # Add some noise for realism
    prob_correct = np.clip(prob_correct + np.random.normal(0, 0.05), 0.1, 0.95)

    is_correct = np.random.random() < prob_correct

    # Time taken (faster students, easier questions = less time)
    base_time = question['estimated_time_sec']
    speed_factor = {'slow': 1.3, 'medium': 1.0, 'fast': 0.8}[student['learning_speed']]
    difficulty_factor = 1 + (question['difficulty'] - 3) * 0.1

    time_taken = base_time * speed_factor * difficulty_factor * np.random.uniform(0.7, 1.3)

    # If wrong, might take more time
    if not is_correct:
        time_taken *= np.random.uniform(1.0, 1.5)

    return {
        'is_correct': int(is_correct),
        'time_taken_sec': round(time_taken),
        'probability': round(prob_correct, 3)
    }

def generate_response_data(questions_df, students, n_responses_per_student=50):
    """Generate student response data"""
    responses = []
    start_date = datetime(2024, 1, 1)

    for student in students:
        # Filter questions for student's class
        class_questions = questions_df[questions_df['class_level'] == student['class_level']]

        # Sample questions for this student
        n_available = len(class_questions)
        n_to_attempt = min(n_responses_per_student, n_available)
        sampled_questions = class_questions.sample(n_to_attempt)

        # Simulate attempts over time
        for i, (_, question) in enumerate(sampled_questions.iterrows()):
            response = simulate_response(student, question)

            # Random timestamp within the last 90 days
            days_offset = np.random.randint(0, 90)
            attempt_time = start_date + timedelta(days=days_offset, hours=np.random.randint(8, 22))

            responses.append({
                'response_id': f'R{len(responses)+1:06d}',
                'student_id': student['student_id'],
                'question_id': question['question_id'],
                'topic': question['topic'],
                'subtopic': question['subtopic'],
                'difficulty': question['difficulty'],
                'is_correct': response['is_correct'],
                'time_taken_sec': response['time_taken_sec'],
                'attempt_timestamp': attempt_time,
                'attempt_number': 1  # Could track multiple attempts
            })

    return pd.DataFrame(responses)

def save_datasets(output_dir='../data'):
    """Generate and save all datasets"""
    os.makedirs(output_dir, exist_ok=True)

    print("Generating question bank...")
    questions_df = generate_question_bank()
    questions_df.to_csv(f'{output_dir}/questions_bank.csv', index=False)
    print(f"Generated {len(questions_df)} questions")

    print("\nGenerating student profiles...")
    students = generate_student_profiles(200)
    student_df = pd.DataFrame([{
        'student_id': s['student_id'],
        'name': s['name'],
        'class_level': s['class_level'],
        'overall_ability': s['overall_ability'],
        'learning_speed': s['learning_speed'],
        'practice_frequency': s['practice_frequency']
    } for s in students])
    student_df.to_csv(f'{output_dir}/students.csv', index=False)
    print(f"Generated {len(students)} student profiles")

    print("\nGenerating response data...")
    responses_df = generate_response_data(questions_df, students, n_responses_per_student=50)
    responses_df.to_csv(f'{output_dir}/student_responses.csv', index=False)
    print(f"Generated {len(responses_df)} response records")

    print(f"\nAll data saved to {output_dir}/")

if __name__ == "__main__":
    save_datasets()
