import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Cricket Match Predictor", page_icon="🏏")
st.title("🏏 IPL Match Outcome Predictor")
st.markdown("Fill in the match details below to predict the winner.")

# Auto-train on cloud if model missing
if not os.path.exists("model.pkl"):
    with st.spinner("Setting up model for first time... this takes ~30 seconds."):
        import subprocess
        result = subprocess.run(["python", "train_model.py"], capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Training failed:\n{result.stderr}")
            st.stop()

try:
    with open("model.pkl", "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    le_team = saved["le_team"]
    le_venue = saved["le_venue"]
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# rest of your app stays exactly the same from here...
teams = list(le_team.classes_)
venues = list(le_venue.classes_)

col1, col2 = st.columns(2)
with col1:
    team1 = st.selectbox("Team 1 (home)", teams)
with col2:
    team2 = st.selectbox("Team 2 (away)", [t for t in teams if t != team1])

venue = st.selectbox("Venue", venues)
toss_winner = st.radio("Toss won by", [team1, team2])
toss_decision = st.radio("Toss decision", ["bat", "field"])

if st.button("Predict winner"):
    toss_bat = 1 if toss_decision == "bat" else 0
    team1_won_toss = 1 if toss_winner == team1 else 0

    team1_enc = le_team.transform([team1])[0]
    team2_enc = le_team.transform([team2])[0]
    venue_enc = le_venue.transform([venue])[0]

    X_input = pd.DataFrame(
        [[team1_enc, team2_enc, venue_enc, toss_bat, team1_won_toss]],
        columns=['team1_enc', 'team2_enc', 'venue_enc', 'toss_bat', 'team1_won_toss']
    )

    prob = model.predict_proba(X_input)[0]
    team1_prob = prob[1]
    team2_prob = prob[0]

    st.divider()
    st.subheader("Prediction")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(team1, f"{team1_prob:.0%}")
    with col_b:
        st.metric(team2, f"{team2_prob:.0%}")

    winner = team1 if team1_prob > 0.5 else team2
    st.success(f"Predicted winner: **{winner}**")
    st.progress(float(team1_prob), text=f"{team1} win probability")