import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import urllib.request
import os

# Download dataset automatically if not present
DATA_URL = "https://raw.githubusercontent.com/dsrscientist/IPL-dataset/master/matches.csv"
DATA_PATH = "matches.csv"

if not os.path.exists(DATA_PATH):
    print("Downloading IPL dataset...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")
        raise

# Load data
df = pd.read_csv(DATA_PATH)

# Drop rows where result is a tie or no result
df = df[df['winner'].notna()]
df = df[df['result'] != 'tie']

# Features
df['toss_bat'] = (df['toss_decision'] == 'bat').astype(int)
df['team1_won_toss'] = (df['toss_winner'] == df['team1']).astype(int)
df['team1_won'] = (df['winner'] == df['team1']).astype(int)

# Encode categorical columns
le_team = LabelEncoder()
le_venue = LabelEncoder()

all_teams = pd.concat([df['team1'], df['team2']])
le_team.fit(all_teams)
le_venue.fit(df['venue'])

df['team1_enc'] = le_team.transform(df['team1'])
df['team2_enc'] = le_team.transform(df['team2'])
df['venue_enc'] = le_venue.transform(df['venue'])

# Select features
features = ['team1_enc', 'team2_enc', 'venue_enc', 'toss_bat', 'team1_won_toss']
X = df[features]
y = df['team1_won']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))

# Save
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "le_team": le_team,
        "le_venue": le_venue
    }, f)

print("model.pkl saved.")