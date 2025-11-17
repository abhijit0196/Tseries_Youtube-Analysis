# YouTube Channel Data Analysis & Views Prediction

## Project Overview
This project performs a comprehensive analysis of a YouTube channel's video data using **Python, Data Science, and Machine Learning**. The goal is to understand video performance trends, engagement patterns, sentiment impact, and predict future views using ML models.

---

## Dataset
The dataset (`new_music_data.csv`) contains the following columns:

- `title` : Video title
- `description` : Video description
- `tags` : Video tags (list)
- `views` : Number of views
- `likes` : Number of likes
- `comments_count` : Number of comments
- `duration` : Video duration (seconds)
- `publishedAt` : Video publish datetime
- `categoryId` : Video category ID

---

## Methodology

### 1. Data Cleaning & Preprocessing
- Checked missing values and data types.
- Converted `publishedAt` to datetime.
- Calculated **engagement rate**: `(likes + comments_count)/views`.
- Applied **log transformations** to skewed numerical features (`views`, `likes`, `comments_count`).
- Extracted features:
  - `title_length`, `description_length`, `tags_count`
  - `publish_weekday`, `publish_hour`
  - `likes_per_view`, `comments_per_view`

### 2. Exploratory Data Analysis (EDA)
- Descriptive statistics and correlation analysis.
- Histograms and boxplots for views, likes, and comments.
- Time series analysis of engagement over time.
- WordCloud visualization for frequent words in titles and descriptions.
- Top 10 tags by average engagement.

### 3. Sentiment Analysis
- Combined titles, descriptions, and tags into single text per video.
- Cleaned text and applied **VADER sentiment analysis**.
- Categorized sentiment into: `positive`, `neutral`, `negative`.
- Visualized sentiment distribution.
- Correlated sentiment with engagement rate.
- Conducted **ANOVA test** to check engagement differences across sentiment categories.

### 4. Machine Learning
- Features used for prediction:
  - `duration`, `likes`, `comments_count`
  - `title_length`, `description_length`, `tags_count`
  - `publish_weekday`, `publish_hour`
  - `likes_per_view`, `comments_per_view`
- Target variable: `views` (both original and log-transformed)
- Models applied:
  - **Random Forest Regressor**
  - **XGBoost Regressor**
  - **Linear Regression**
- Evaluation Metrics:
  - RMSE (Root Mean Squared Error)
  - R² Score
- Feature importance analyzed using Random Forest.

---

## Key Insights
- Engagement rate varies by video length, tags, and publish timing.
- Certain tags drive higher engagement.
- Positive sentiment videos tend to have slightly higher engagement.
- Log transformation improves ML model performance on skewed views data.
- Top features impacting views:
  - Likes
  - Comments
  - Duration
  - Title length
  - Description length

---

## Visualizations
- Histograms & boxplots for views, likes, comments.
- Heatmaps for correlation.
- Time series plots for engagement rate over time.
- WordClouds for frequent words in titles and descriptions.
- Bar charts for top tags by engagement.
- Feature importance plots for ML models.
- Sentiment vs engagement boxplots.

---

## Technologies Used
- Python Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`, `nltk`
- Machine Learning: `scikit-learn` (Random Forest, Linear Regression), `xgboost`
- Statistical Analysis: `scipy.stats` (T-test, ANOVA)

---

## Usage
1. Clone the repository.
2. Place `new_music_data.csv` in the project folder.
3. Run the Jupyter Notebook (`.ipynb`) step by step to reproduce analysis, visualizations, and ML models.
4. Modify features or models for experimentation.

---

## Author
- Data Science & ML Project by Abhijit Wabale
