# IPL Match Data Analysis: Team Performance and Winning Patterns

Project to analyze historical Indian Premier League (IPL) match data and visualize team performance, toss impact, venue effects, player awards, and season-wise trends using Streamlit and Plotly.

**Project Structure**

IPL_Analysis_Project/

- data/
  - matches.csv <- Place the Kaggle `matches.csv` here
- app.py
- analysis.py
- requirements.txt
- README.md

Note: The project focuses on data analysis and visualizations. (Modeling UI removed.)

UI / Visual design
- The Streamlit app uses a subtle two-tone background and a vibrant gradient header for visual focus.
- Section content is presented in white "cards" with rounded corners and soft shadows for improved readability.
- Colors: deep-teal sidebar gradient and a cyan→violet header gradient were chosen for high contrast and modern appearance. You can customize these in `app.py` (CSS block at the top of the file).

Where to place dataset

- Download `matches.csv` from the Kaggle IPL dataset and place it at `IPL_Analysis_Project/data/matches.csv`.

Installation (Windows)

1. Create a virtual environment and activate it

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run the Streamlit app

```powershell
streamlit run app.py
```

What the app shows

- Team Performance: Total matches, wins and win percentage per team
- Toss Analysis: Impact of toss winner on match outcome
- Venue Analysis: Which stadiums favor which teams
- Player Awards: Most Player of the Match awards
- Season Trends: Top team per season (by wins)
- ML Prediction Lab:
  - Trains 3 algorithms: Logistic Regression, Random Forest, Gradient Boosting
  - Shows 5 graphs:
    1) Model metric comparison (bar)
    2) Confusion matrix (heatmap)
    3) Feature importance (bar)
    4) Confidence distribution (histogram)
    5) Predicted winner share (pie)
  - Includes custom match winner prediction form

Data preprocessing details (in `analysis.py`)

- Loads `data/matches.csv` using pandas
- Drops rows missing critical fields
- Converts the `date` column to datetime and creates `match_year`
- Removes unused columns like `id`, `umpire1`, `umpire2`, `umpire3`
- Standardizes team names (mapping legacy names to current ones)
- Creates `win_margin` (from runs or wickets) and `toss_win_match_win` flag

Notes

- This project uses only Python, Pandas, NumPy, Plotly and Streamlit as requested.
- Optional logistic regression model was not added to keep the stack minimal and follow the requested libraries.

Viva / Explanation pointers

- Explain data cleaning steps: NA handling, date parsing, team standardization.
- Explain each visualization and what it reveals about team advantage, toss influence, and venues.
- Discuss limitations: winner labels rely on the CSV accuracy; mapping of team names is simple and may need expansion.
