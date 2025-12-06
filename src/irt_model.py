"""
Item Response Theory (IRT) Model for Adaptive Learning
Estimates student ability and predicts success probability
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

class IRTModel:
    """2-Parameter Logistic IRT Model"""

    def __init__(self, discrimination=1.5):
        self.discrimination = discrimination  # 'a' parameter
        self.question_difficulties = {}  # 'b' parameter per question
        self.student_abilities = {}  # theta per student

    def probability_correct(self, ability, difficulty):
        """Calculate probability of correct response using 2PL model"""
        logit = self.discrimination * (ability - difficulty)
        return 1 / (1 + np.exp(-logit))

    def estimate_difficulty(self, responses_df):
        """Estimate question difficulty from response data"""
        for question_id in responses_df['question_id'].unique():
            q_responses = responses_df[responses_df['question_id'] == question_id]
            # Simple estimate: logit of proportion correct
            prop_correct = q_responses['is_correct'].mean()
            # Bound to avoid extreme values
            prop_correct = np.clip(prop_correct, 0.05, 0.95)
            # Convert to difficulty scale
            difficulty = -np.log(prop_correct / (1 - prop_correct)) / self.discrimination
            self.question_difficulties[question_id] = difficulty

    def estimate_ability(self, responses_df):
        """Estimate student ability from response data"""
        for student_id in responses_df['student_id'].unique():
            s_responses = responses_df[responses_df['student_id'] == student_id]

            if len(s_responses) < 3:
                self.student_abilities[student_id] = 0  # Default average ability
                continue

            # MLE estimation of ability
            def neg_log_likelihood(theta):
                ll = 0
                for _, row in s_responses.iterrows():
                    difficulty = self.question_difficulties.get(row['question_id'], 0)
                    p = self.probability_correct(theta[0], difficulty)
                    p = np.clip(p, 1e-10, 1 - 1e-10)
                    if row['is_correct']:
                        ll += np.log(p)
                    else:
                        ll += np.log(1 - p)
                return -ll

            result = minimize(neg_log_likelihood, [0], method='L-BFGS-B', bounds=[(-3, 3)])
            self.student_abilities[student_id] = result.x[0]

    def fit(self, responses_df):
        """Fit IRT model to response data"""
        print("Estimating question difficulties...")
        self.estimate_difficulty(responses_df)

        print("Estimating student abilities...")
        self.estimate_ability(responses_df)

        return self

    def predict_probability(self, student_id, question_id):
        """Predict probability of correct response"""
        ability = self.student_abilities.get(student_id, 0)
        difficulty = self.question_difficulties.get(question_id, 0)
        return self.probability_correct(ability, difficulty)

    def evaluate(self, responses_df):
        """Evaluate model performance"""
        y_true = []
        y_pred_prob = []
        y_pred = []

        for _, row in responses_df.iterrows():
            prob = self.predict_probability(row['student_id'], row['question_id'])
            y_true.append(row['is_correct'])
            y_pred_prob.append(prob)
            y_pred.append(1 if prob > 0.5 else 0)

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_prob)

        return {'accuracy': accuracy, 'auc': auc}


class KnowledgeTracer:
    """Track student knowledge state per topic"""

    def __init__(self):
        self.topic_mastery = {}  # student_id -> topic -> mastery score

    def update_mastery(self, responses_df):
        """Calculate topic mastery scores"""
        for student_id in responses_df['student_id'].unique():
            student_responses = responses_df[responses_df['student_id'] == student_id]

            self.topic_mastery[student_id] = {}

            for topic in student_responses['topic'].unique():
                topic_responses = student_responses[student_responses['topic'] == topic]

                if len(topic_responses) < 2:
                    continue

                # Calculate weighted accuracy (recent attempts matter more)
                topic_responses = topic_responses.sort_values('attempt_timestamp')
                weights = np.linspace(0.5, 1.0, len(topic_responses))
                weighted_accuracy = np.average(topic_responses['is_correct'], weights=weights)

                # Consider difficulty
                avg_difficulty = topic_responses['difficulty'].mean()
                difficulty_adjusted = weighted_accuracy * (1 + (avg_difficulty - 3) * 0.1)

                self.topic_mastery[student_id][topic] = np.clip(difficulty_adjusted, 0, 1)

    def get_weak_topics(self, student_id, threshold=0.6):
        """Get topics where student needs improvement"""
        if student_id not in self.topic_mastery:
            return []

        mastery = self.topic_mastery[student_id]
        weak = [(topic, score) for topic, score in mastery.items() if score < threshold]
        return sorted(weak, key=lambda x: x[1])

    def get_strong_topics(self, student_id, threshold=0.8):
        """Get topics where student is strong"""
        if student_id not in self.topic_mastery:
            return []

        mastery = self.topic_mastery[student_id]
        strong = [(topic, score) for topic, score in mastery.items() if score >= threshold]
        return sorted(strong, key=lambda x: x[1], reverse=True)


class QuestionRecommender:
    """Recommend next questions based on student state"""

    def __init__(self, irt_model, knowledge_tracer, questions_df):
        self.irt = irt_model
        self.kt = knowledge_tracer
        self.questions = questions_df

    def recommend_next(self, student_id, n_questions=5, strategy='adaptive'):
        """Recommend next questions for a student"""
        student_ability = self.irt.student_abilities.get(student_id, 0)
        weak_topics = [t for t, _ in self.kt.get_weak_topics(student_id)]

        recommendations = []

        if strategy == 'adaptive':
            # Mix of weak topic practice and zone of proximal development
            # Target questions where P(correct) is 0.5-0.7
            for _, q in self.questions.iterrows():
                prob = self.irt.predict_probability(student_id, q['question_id'])
                priority = 0

                # Prioritize weak topics
                if q['topic'] in weak_topics:
                    priority += 2

                # Optimal difficulty zone
                if 0.4 < prob < 0.7:
                    priority += 3
                elif 0.3 < prob < 0.8:
                    priority += 1

                recommendations.append({
                    'question_id': q['question_id'],
                    'topic': q['topic'],
                    'subtopic': q['subtopic'],
                    'difficulty': q['difficulty'],
                    'success_probability': round(prob, 2),
                    'priority': priority
                })

        elif strategy == 'remedial':
            # Focus on weak topics with easier questions
            for _, q in self.questions.iterrows():
                if q['topic'] in weak_topics and q['difficulty'] <= 3:
                    prob = self.irt.predict_probability(student_id, q['question_id'])
                    recommendations.append({
                        'question_id': q['question_id'],
                        'topic': q['topic'],
                        'subtopic': q['subtopic'],
                        'difficulty': q['difficulty'],
                        'success_probability': round(prob, 2),
                        'priority': 5 - q['difficulty']
                    })

        # Sort by priority and return top N
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        return recommendations[:n_questions]


def main():
    """Run IRT model training"""
    print("="*60)
    print("CBSE ADAPTIVE PRACTICE ENGINE - IRT MODEL")
    print("="*60)

    # Load data
    questions_df = pd.read_csv('../data/questions_bank.csv')
    responses_df = pd.read_csv('../data/student_responses.csv')

    print(f"Questions: {len(questions_df)}")
    print(f"Responses: {len(responses_df)}")

    # Train IRT model
    irt = IRTModel(discrimination=1.5)
    irt.fit(responses_df)

    # Evaluate
    metrics = irt.evaluate(responses_df)
    print(f"\nModel Performance:")
    print(f"  Accuracy: {metrics['accuracy']*100:.1f}%")
    print(f"  AUC: {metrics['auc']:.3f}")

    # Knowledge tracing
    kt = KnowledgeTracer()
    kt.update_mastery(responses_df)

    # Sample student analysis
    sample_student = responses_df['student_id'].iloc[0]
    print(f"\nSample Student Analysis: {sample_student}")
    print(f"  Ability: {irt.student_abilities.get(sample_student, 0):.2f}")
    print(f"  Weak Topics: {kt.get_weak_topics(sample_student)}")
    print(f"  Strong Topics: {kt.get_strong_topics(sample_student)}")

    # Recommendations
    recommender = QuestionRecommender(irt, kt, questions_df)
    recommendations = recommender.recommend_next(sample_student, n_questions=5)
    print(f"\nRecommended Questions:")
    for r in recommendations:
        print(f"  {r['question_id']}: {r['topic']} (Difficulty: {r['difficulty']}, P(correct): {r['success_probability']})")

    # Save models
    os.makedirs('../models', exist_ok=True)
    joblib.dump(irt, '../models/irt_model.joblib')
    joblib.dump(kt, '../models/knowledge_tracer.joblib')
    print("\nModels saved to ../models/")

if __name__ == "__main__":
    main()
