"""
Student Dashboard for CBSE Adaptive Practice Engine
Personalized learning interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="CBSE Practice Engine", page_icon="📚", layout="wide")

st.title("📚 CBSE Adaptive Practice Engine")
st.markdown("Personalized practice for Class 9-10 Biology")

@st.cache_data
def load_data():
    try:
        questions = pd.read_csv('../data/questions_bank.csv')
        responses = pd.read_csv('../data/student_responses.csv')
        students = pd.read_csv('../data/students.csv')
        return questions, responses, students
    except:
        return None, None, None

questions, responses, students = load_data()

# Sidebar - Student selection
st.sidebar.header("Student Profile")
if students is not None:
    student_id = st.sidebar.selectbox("Select Student", students['student_id'].unique())
    student_info = students[students['student_id'] == student_id].iloc[0]
    st.sidebar.write(f"**Class:** {student_info['class_level'].replace('_', ' ').title()}")
    st.sidebar.write(f"**Ability:** {student_info['overall_ability']:.2f}")
    st.sidebar.write(f"**Learning Speed:** {student_info['learning_speed'].title()}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📝 Practice", "📈 Progress", "🎯 Recommendations"])

with tab1:
    st.header("Your Learning Dashboard")

    if responses is not None and student_id:
        student_responses = responses[responses['student_id'] == student_id]

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Questions Attempted", len(student_responses))
        with col2:
            accuracy = student_responses['is_correct'].mean() * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        with col3:
            avg_time = student_responses['time_taken_sec'].mean()
            st.metric("Avg Time/Question", f"{avg_time:.0f}s")
        with col4:
            topics_covered = student_responses['topic'].nunique()
            st.metric("Topics Covered", topics_covered)

        # Topic-wise performance
        st.subheader("Performance by Topic")
        topic_perf = student_responses.groupby('topic').agg({
            'is_correct': ['mean', 'count']
        }).reset_index()
        topic_perf.columns = ['Topic', 'Accuracy', 'Attempts']
        topic_perf['Accuracy'] = topic_perf['Accuracy'] * 100

        fig = px.bar(topic_perf, x='Topic', y='Accuracy', color='Accuracy',
                    color_continuous_scale='RdYlGn', range_color=[0, 100])
        fig.add_hline(y=70, line_dash="dash", line_color="gray", annotation_text="Target: 70%")
        st.plotly_chart(fig, use_container_width=True)

        # Difficulty breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Performance by Difficulty")
            diff_perf = student_responses.groupby('difficulty')['is_correct'].mean() * 100
            fig = px.bar(x=diff_perf.index, y=diff_perf.values,
                        labels={'x': 'Difficulty Level', 'y': 'Accuracy %'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Time vs Accuracy")
            fig = px.scatter(student_responses, x='time_taken_sec', y='is_correct',
                           color='difficulty', size='difficulty',
                           labels={'time_taken_sec': 'Time (sec)', 'is_correct': 'Correct'})
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Practice Questions")

    if questions is not None and student_id:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_topic = st.selectbox("Topic", ['All'] + list(questions['topic'].unique()))
        with col2:
            difficulty_range = st.slider("Difficulty", 1, 5, (1, 5))
        with col3:
            q_type = st.selectbox("Question Type", ['All', 'MCQ', 'True/False', 'Short Answer'])

        # Filter questions
        filtered_q = questions.copy()
        if selected_topic != 'All':
            filtered_q = filtered_q[filtered_q['topic'] == selected_topic]
        filtered_q = filtered_q[
            (filtered_q['difficulty'] >= difficulty_range[0]) &
            (filtered_q['difficulty'] <= difficulty_range[1])
        ]
        if q_type != 'All':
            filtered_q = filtered_q[filtered_q['question_type'] == q_type]

        st.write(f"**Available Questions:** {len(filtered_q)}")

        if st.button("Start Practice Session", type="primary"):
            # Sample 5 questions
            practice_qs = filtered_q.sample(min(5, len(filtered_q)))

            for i, (_, q) in enumerate(practice_qs.iterrows()):
                with st.expander(f"Question {i+1}: {q['subtopic']} (Difficulty: {q['difficulty']})"):
                    st.write(f"**Topic:** {q['topic']}")
                    st.write(f"**Type:** {q['question_type']}")
                    st.write(f"**Estimated Time:** {q['estimated_time_sec']} seconds")
                    st.write(f"**Question ID:** {q['question_id']}")

                    if q['question_type'] == 'MCQ':
                        st.radio("Your Answer:", ['A', 'B', 'C', 'D'], key=f"q_{q['question_id']}")
                    elif q['question_type'] == 'True/False':
                        st.radio("Your Answer:", ['True', 'False'], key=f"q_{q['question_id']}")
                    else:
                        st.text_area("Your Answer:", key=f"q_{q['question_id']}")

with tab3:
    st.header("Learning Progress")

    if responses is not None and student_id:
        student_responses = responses[responses['student_id'] == student_id].copy()
        student_responses['attempt_timestamp'] = pd.to_datetime(student_responses['attempt_timestamp'])

        # Progress over time
        st.subheader("Accuracy Trend Over Time")
        daily_perf = student_responses.groupby(student_responses['attempt_timestamp'].dt.date).agg({
            'is_correct': 'mean'
        }).reset_index()
        daily_perf.columns = ['Date', 'Accuracy']
        daily_perf['Accuracy'] *= 100

        fig = px.line(daily_perf, x='Date', y='Accuracy', markers=True)
        fig.add_hline(y=70, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        # Topic mastery radar chart
        st.subheader("Topic Mastery")
        topic_mastery = student_responses.groupby('topic')['is_correct'].mean() * 100
        fig = go.Figure(go.Scatterpolar(
            r=topic_mastery.values,
            theta=topic_mastery.index,
            fill='toself'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Personalized Recommendations")

    if responses is not None and student_id:
        student_responses = responses[responses['student_id'] == student_id]

        # Identify weak topics
        topic_perf = student_responses.groupby('topic')['is_correct'].mean()
        weak_topics = topic_perf[topic_perf < 0.6].sort_values()

        st.subheader("Areas for Improvement")
        if not weak_topics.empty:
            for topic, score in weak_topics.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📍 **{topic.replace('_', ' ').title()}**")
                with col2:
                    st.write(f"Accuracy: {score*100:.0f}%")
        else:
            st.success("Great job! You're performing well in all topics!")

        # Recommended questions
        st.subheader("Recommended Practice Questions")
        if not weak_topics.empty:
            weak_topic_list = list(weak_topics.index)
            recommended = questions[
                (questions['topic'].isin(weak_topic_list)) &
                (questions['difficulty'] <= 3)
            ].sample(min(5, len(questions)))

            for _, q in recommended.iterrows():
                st.write(f"- {q['question_id']}: {q['subtopic']} ({q['topic']}) - Difficulty: {q['difficulty']}")

        # Study plan
        st.subheader("Suggested Study Plan")
        st.info("""
        Based on your performance, we recommend:
        1. **Focus on weak topics** - Spend extra time on topics with <60% accuracy
        2. **Practice daily** - Aim for 10-15 questions per day
        3. **Start easy** - Begin with difficulty 1-2 questions before advancing
        4. **Review mistakes** - Understand why you got questions wrong
        """)

st.markdown("---")
st.markdown("**CBSE Adaptive Practice Engine** | Personalized Learning for Indian Students | © 2024")
