# MLB Game Prediction App

### ✨ Last Updated: July 31, 2025

---

## 🌟 Project Overview

This project builds a machine learning pipeline and Streamlit UI to predict the outcome of MLB games based on team-level stats using data from the 2025 season.

### Features:

* Predict all **upcoming MLB games** using trained models
* View **past predictions** and historical outcomes
* Navigate by date to see predictions from **any day of the season**
* Stores and reuses predictions to improve future model accuracy

---

## 📊 Data Source

Data is collected from the **official MLB Stats API** and includes:

* Full 2025 season schedule
* Game results
* Team-level boxscore stats (batting/pitching)

---

## 📁 Files and Scripts

### 1. `db_setup.py`

Initializes and populates the database:

* Creates SQLite tables
* Fetches full 2025 schedule and game results
* Ingests team stats
* Saves everything to `mlb_predictions.db`

**Run this first**:

```bash
python db_setup.py
```

---

### 2. `predict_today.py`

Trains models and makes predictions for **today's games**:

* Updates yesterday's results
* Recomputes team stats
* Trains models using current data
* Saves today's predictions into the DB

**Run this daily** (can be automated with cron):

```bash
python predict_today.py
```

---

### 3. `app.py`

Streamlit user interface for interacting with predictions:

* Shows stored predictions for today
* Allows batch prediction of **future games**
* Dropdown menu to browse **historical predictions**

**Launch UI**:

```bash
streamlit run app.py
```

---

### 4. `mlb_feature_engineering.py`

Machine learning pipeline:

* Extracts per-game boxscore features
* Imputes missing values
* Trains 3 models:

  * Classifier for win probability
  * Regressor for margin
  * Regressor for total runs

Also includes a helper function to generate features for future games.

---

## 🤝 Recommended Daily Workflow

1. Update the database:

```bash
python db_setup.py
```

2. Run prediction pipeline for today's games:

```bash
python predict_today.py
```

3. Launch the app:

```bash
streamlit run app.py
```

---

## 🤔 FAQ

**Q: What data is used to make predictions?**
A: Team-level stats from each game, stored and retrieved from the SQLite DB.

**Q: Are predictions saved?**
A: Yes, predictions for each date are saved and reused for performance and future training.

**Q: Can we extend this model to player-level stats?**
A: Yes! Once the team-level model is stable, we can integrate deeper stat layers.

---

## 🚀 Next Steps

* Automate daily updates with a `cron` or scheduled task
* Add player-level metrics (future enhancement)
* Display model confidence intervals and visual trends in Streamlit

---

## 📚 License

This project is for educational and predictive modeling purposes. All MLB data belongs to its respective owners.
