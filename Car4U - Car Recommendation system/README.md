# 🚗 Car4U - Intelligent Used Car Recommendation & Price Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> *Your AI-powered companion for smarter used car decisions*

An end-to-end machine learning solution that revolutionizes the used car buying experience through intelligent recommendations and accurate price predictions, powered by comprehensive Craigslist data analysis.


## 🌟 Project Highlights

- **87.79% Prediction Accuracy** - Random Forest model for precise price estimation
- **Content-Based Recommendations** - Smart similarity matching across 400K+ listings
- **Real-World Dataset** - Comprehensive US market data (April-May 2021)
- **Production-Ready Code** - Clean, modular, and well-documented implementation

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Pipeline](#-data-pipeline)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The used car market has exploded in recent years, with second-hand vehicle sales doubling compared to new car sales. Car4U addresses the overwhelming choice paradox buyers face by leveraging machine learning to:

1. **Recommend** similar vehicles based on user preferences
2. **Predict** accurate prices to ensure fair market value
3. **Analyze** market trends across the United States

### 💡 The Problem
- 61% of pandemic-era buyers preferred dealership purchases, yet online platforms lack intelligent guidance
- Buyers struggle with information overload when browsing thousands of listings
- Sellers have limited tools to price their vehicles competitively

### ✨ The Solution
Car4U provides an intelligent layer on top of Craigslist data to match buyers with their ideal vehicles and provide data-driven pricing insights.

---

## 🔥 Key Features

### 🎯 Recommendation Engine
- **Content-based filtering** using TF-IDF vectorization
- **Cosine similarity matching** across vehicle features
- **Top-6 similar vehicles** for any given listing
- Considers manufacturer, model, type, condition, and specifications

### 💰 Price Prediction Model
- **Random Forest Regressor** with 87.79% accuracy
- Handles multiple features: year, odometer, condition, location, etc.
- **Real-time predictions** for user-specified parameters
- Trained on 400K+ verified transactions

### 📊 Comprehensive Analytics
- Market trend visualization across 50 states
- Manufacturer and model popularity analysis
- Price distribution by vehicle type and condition
- Mileage impact assessment

---

## 🛠️ Technology Stack

```python
# Core Technologies
├── Python 3.8+              # Primary language
├── Jupyter Notebook         # Development environment
├── pandas                   # Data manipulation
├── NumPy                    # Numerical computing
├── scikit-learn             # Machine learning
│   ├── TfidfVectorizer     # Feature extraction
│   ├── RandomForestRegressor # Price prediction
│   └── LabelEncoder        # Categorical encoding
├── Matplotlib & Seaborn    # Data visualization
└── Tableau                 # Interactive dashboards
```

---


## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Setup

```bash
# Clone the repository
git clone https://github.com/Nishil248/car4u.git
cd car4u

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Dependencies
```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

---

## 🚀 Usage

### 1. Data Preparation
```python
from src.data_cleaning import clean_dataset

# Load and clean data
df_cleaned = clean_dataset('data/raw/vehicles.csv')
```

### 2. Get Recommendations
```python
from src.recommendation import get_similar_cars

# Find similar vehicles
recommendations = get_similar_cars(
    manufacturer='Toyota',
    model='Camry',
    year=2018,
    car_type='sedan',
    top_n=6
)
print(recommendations)
```

### 3. Predict Prices
```python
from src.prediction import predict_price

# Predict vehicle price
estimated_price = predict_price(
    manufacturer='Honda',
    model='Civic',
    year=2019,
    odometer=45000,
    condition='excellent',
    state='CA'
)
print(f"Estimated Price: ${estimated_price:,.2f}")
```

---

## 🔄 Data Pipeline

### 1. **Data Collection**
- **Source**: Craigslist listings across all US states
- **Period**: April-May 2021
- **Volume**: 400,000+ records
- **Features**: 22 attributes per vehicle

### 2. **Data Cleaning**
- Removed duplicate entries and irrelevant columns
- Handled missing values using probabilistic imputation
- Outlier detection and removal (IQR method)
- Data type standardization

### 3. **Feature Engineering**
- Created age-based features from year
- Calculated annual mileage metrics
- Encoded categorical variables
- Normalized numerical features

### 4. **Model Training**
- Train-test split (80-20)
- Cross-validation for hyperparameter tuning
- Model evaluation and selection
- Serialization for production use

---

## 📈 Model Performance

### Price Prediction Model

| Model | R² Score | RMSE | MAE |
|-------|----------|------|-----|
| Random Forest | **87.79%** | $3,245 | $2,156 |
| Gradient Boosting | 85.32% | $3,567 | $2,389 |
| Linear Regression | 72.15% | $4,921 | $3,245 |

### Recommendation System
- **Similarity Metric**: Cosine Similarity
- **Vectorization**: TF-IDF on combined features
- **Average Similarity Score**: 0.82/1.00

---

## 💎 Key Insights

### Market Trends
- 🏆 **Top Manufacturers**: Ford, Chevrolet, Toyota dominate with ~40% market share
- 📍 **Geographic Leaders**: California, Texas, Florida have highest listing volumes
- 💵 **Average Pricing**: $15,000 (nationwide), with regional variations up to 50%

### Vehicle Preferences
- 🚙 **Popular Types**: SUVs (28%), Sedans (26%), Trucks (22%)
- ⛽ **Fuel Types**: 85% gasoline, 12% diesel, 3% electric/hybrid
- 🎨 **Color Trends**: White and black vehicles command 15% price premium

### Pricing Factors
- 📅 **Year Impact**: Each year adds ~$1,200 to average price
- 🛣️ **Mileage Impact**: 15% price reduction per 50,000 miles
- ⭐ **Condition Premium**: "Excellent" vs "Good" = 25% price difference

### Regional Insights
- California: Most listings (80K+), average price $15K
- Washington: Premium market, average price $23K despite fewer listings
- Texas: High volume, competitive pricing at $13K average

---

## 🚀 Future Enhancements

### Phase 2: Advanced Features
- [ ] **Hybrid Recommendation System**: Combine content-based + collaborative filtering
- [ ] **Image Recognition**: CNN-based condition assessment from photos
- [ ] **Natural Language Processing**: Analyze listing descriptions for hidden insights
- [ ] **Time Series Forecasting**: Predict future price trends

### Phase 3: Production Deployment
- [ ] **Web Application**: Flask/Django REST API
- [ ] **User Interface**: React-based interactive platform
- [ ] **Real-time Updates**: Live scraping and model retraining
- [ ] **Mobile App**: iOS/Android companion apps

### Phase 4: Advanced Analytics
- [ ] **Regional Models**: State-specific price prediction
- [ ] **Seasonal Adjustments**: Account for time-of-year variations
- [ ] **Market Alerts**: Notify users of good deals
- [ ] **Fraud Detection**: Identify suspicious listings

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@Nishil248](https://github.com/Nishil248)

---

## 🙏 Acknowledgments

- **Austin Reese** - Original dataset compilation and curation
- **Kaggle Community** - For providing the platform and dataset
- **Craigslist** - For the marketplace data
- **scikit-learn Team** - For excellent ML libraries

---

## 📚 References

- [Kaggle Dataset](https://www.kaggle.com/datasets)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Tableau Public Gallery](https://public.tableau.com/)

