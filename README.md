# 📚 Adaptive CBSE Practice Engine (EdTech)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ML](https://img.shields.io/badge/ML-IRT%20|%20Knowledge%20Tracing-orange.svg)
![EdTech](https://img.shields.io/badge/Domain-EdTech-purple.svg)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An adaptive practice platform for **CBSE Classes 9-10 Biology** using **Item Response Theory (IRT)** and ML-based personalized learning. Features knowledge tracing, next-question prediction, and personalized practice paths.

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [IRT Model](#-irt-model)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Knowledge Tracing](#-knowledge-tracing)
- [Dashboard Features](#-dashboard-features)
- [Future Enhancements](#-future-enhancements)
- [Author](#-author)

---

## 🎯 Overview

Traditional practice systems don't adapt to individual student needs. This platform provides:

- ✅ **Item Response Theory (IRT)** for ability estimation
- ✅ **Knowledge tracing** for weak topic identification
- ✅ **82% accuracy** in next-question prediction
- ✅ **Personalized practice paths**
- ✅ **Student & instructor dashboards**

### 📊 Platform Statistics
| Metric | Value |
|--------|-------|
| Questions | **300+** |
| Topics Covered | **10** |
| Prediction Accuracy | **82%** |
| Students Simulated | **200** |

---

## ✨ Key Features

### 🧠 Item Response Theory (IRT)
2-Parameter Logistic Model:
```
P(correct) = 1 / (1 + exp(-a × (θ - b)))

Where:
├── θ = Student ability (estimated)
├── b = Question difficulty (calibrated)
└── a = Discrimination parameter (1.5)
```

### 📈 Knowledge Tracing
- Topic-wise mastery tracking
- Weighted recent performance
- Difficulty-adjusted scores
- Weak topic identification

### 🎯 Smart Recommendations
- Zone of Proximal Development targeting
- Weak topic prioritization
- Adaptive difficulty progression
- Remedial vs challenge modes

### 📊 Dual Dashboards
- **Student**: Progress, weak areas, recommendations
- **Instructor**: Class analytics, intervention alerts

---

## 📚 CBSE Biology Topics

### Class 9
| Topic | Subtopics |
|-------|-----------|
| Cell Biology | Structure, Organelles, Division |
| Tissues | Plant, Animal, Functions |
| Diversity | Classification, Kingdoms |
| Health & Diseases | Infectious, Prevention |
| Natural Resources | Soil, Water, Air |

### Class 10
| Topic | Subtopics |
|-------|-----------|
| Life Processes | Nutrition, Respiration, Transport |
| Reproduction | Asexual, Sexual, Human |
| Heredity & Evolution | Mendel, Genetics, Natural Selection |
| Environment | Ecosystem, Food Chain, Biodiversity |

---

## 🧮 IRT Model

### Ability Estimation (MLE)
```python
# Maximum Likelihood Estimation
def estimate_ability(responses):
    θ = optimize.minimize(
        neg_log_likelihood,
        x0=[0],  # Initial guess
        bounds=[(-3, 3)]  # Ability range
    )
    return θ
```

### Difficulty Calibration
```
Difficulty = -log(p / (1-p)) / a

Where p = proportion correct for question
```

### Model Performance
| Metric | Value |
|--------|-------|
| Accuracy | 82% |
| AUC-ROC | 0.87 |
| Correlation with actual | 0.84 |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Core ML** | Scikit-learn, SciPy |
| **Data** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **Dashboard** | Streamlit |
| **Serialization** | Joblib |

---

## 📁 Project Structure

```
06_CBSE_Adaptive_Engine/
│
├── 📂 data/
│   ├── questions_bank.csv           # Question bank with metadata
│   ├── student_responses.csv        # Response history
│   └── students.csv                 # Student profiles
│
├── 📂 src/
│   ├── data_generator.py            # Synthetic data generation
│   ├── irt_model.py                 # IRT implementation
│   ├── knowledge_tracer.py          # Knowledge state tracking
│   └── recommender.py               # Question recommendation
│
├── 📂 dashboard/
│   ├── student_app.py               # Student-facing dashboard
│   └── instructor_app.py            # Teacher analytics
│
├── 📂 models/
│   ├── irt_model.joblib             # Trained IRT model
│   └── knowledge_tracer.joblib      # Knowledge state model
│
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/Abin544/cbse-adaptive-engine.git
cd cbse-adaptive-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📖 Usage

### Step 1: Generate Data
```bash
cd src
python data_generator.py
```
Creates question bank and simulated student responses.

### Step 2: Train IRT Model
```bash
python irt_model.py
```

### Step 3: Launch Student Dashboard
```bash
cd ../dashboard
streamlit run student_app.py
```

---

## 📊 Knowledge Tracing

### Topic Mastery Calculation
```python
mastery = weighted_accuracy × (1 + difficulty_bonus)

Where:
├── weighted_accuracy = recent attempts weighted higher
└── difficulty_bonus = adjustment for question difficulty
```

### Sample Student Analysis
```
Student: STU_0042
Overall Ability: 0.72 (Above Average)

Strong Topics (>80%):
├── Cell Biology: 89%
├── Natural Resources: 85%
└── Tissues: 82%

Weak Topics (<60%):
├── Heredity & Evolution: 48%
└── Reproduction: 55%

Recommendation: Focus on Genetics and Mendel's Laws
Next 5 Questions: Q0234, Q0187, Q0156, Q0298, Q0312
```

---

## 🖥️ Dashboard Features

### Student Dashboard
| Section | Features |
|---------|----------|
| **Overview** | Accuracy, attempts, progress |
| **Practice** | Filtered question selection |
| **Progress** | Trend charts, topic radar |
| **Recommendations** | Personalized next steps |

### Instructor Dashboard
| Section | Features |
|---------|----------|
| **Class Overview** | Aggregate performance |
| **Student Details** | Individual analysis |
| **Topic Analysis** | Class-wide weak areas |
| **Intervention Alerts** | At-risk students |

---

## 📈 Learning Path Example

```
Session 1 (Baseline):
├── Mixed difficulty questions
├── Ability estimate: 0.45
└── Identified weak: Heredity

Session 2 (Remedial):
├── Easy Heredity questions
├── 70% accuracy achieved
└── Ability updated: 0.52

Session 3 (Progressive):
├── Medium Heredity questions
├── 65% accuracy
└── Ability: 0.58

Session 4 (Challenge):
├── Mixed topics, higher difficulty
├── Targeting Zone of Proximal Development
└── Continuous adaptation
```

---

## 🔮 Future Enhancements

- [ ] Multi-subject support (Physics, Chemistry, Math)
- [ ] Gamification elements
- [ ] Spaced repetition integration
- [ ] Video explanation links
- [ ] Parent dashboard
- [ ] Mobile app (React Native)
- [ ] Integration with school LMS
- [ ] Voice-based practice

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Adding more subjects
- Question bank expansion
- Advanced IRT models (3PL, MIRT)
- UI/UX improvements

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**Shiva Krupa Abinash Sahu**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/shiva-krupa-abinash-sahu-211692193/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/Abin544)
[![Email](https://img.shields.io/badge/Email-Contact-red?style=flat&logo=gmail)](mailto:abinash.sahu.147@gmail.com)

---

## 🙏 Acknowledgments

- CBSE curriculum for topic structure
- EdTech research papers on adaptive learning
- Item Response Theory literature

---

⭐ **Star this repo if you found it helpful!**
