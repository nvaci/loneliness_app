import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

data = [
    # Loneliness
    ("I reached out, but no one responded.", "loneliness"),
    ("Every attempt to connect ends in silence.", "loneliness"),
    ("I wandered through the game, unseen and unheard.", "loneliness"),
    ("The emptiness was palpable; I was alone.", "loneliness"),
    ("No matter where I went, I remained solitary.", "loneliness"),
    ("I felt like a ghost passing through others.", "loneliness"),
    ("Isolation was my only companion.", "loneliness"),
    ("I existed, but no one acknowledged me.", "loneliness"),
    ("The world was populated, yet I was alone.", "loneliness"),
    ("I tried to engage, but was always ignored.", "loneliness"),
    ("Being surrounded didn't alleviate my loneliness.", "loneliness"),
    ("I moved through crowds, yet felt invisible.", "loneliness"),
    ("My presence seemed to repel others.", "loneliness"),
    ("I was a solitary figure in a busy world.", "loneliness"),
    ("Attempts at connection were met with avoidance.", "loneliness"),
    ("I felt like an outcast in every scene.", "loneliness"),
    ("The game mirrored my feelings of being alone.", "loneliness"),
    ("No matter how I tried, I couldn't find companionship.", "loneliness"),
    ("I was present, but perpetually alone.", "loneliness"),
    ("Loneliness was the game's constant theme.", "loneliness"),

    # Sadness
    ("The game's atmosphere left me feeling downcast.", "sadness"),
    ("A deep sadness settled in as I played.", "sadness"),
    ("I couldn't shake the melancholic mood.", "sadness"),
    ("The experience evoked a profound sorrow.", "sadness"),
    ("I felt a lingering sadness after playing.", "sadness"),
    ("The game's tone was somber and affecting.", "sadness"),
    ("I was moved to tears by the narrative.", "sadness"),
    ("A sense of grief permeated the gameplay.", "sadness"),
    ("The story touched on painful emotions.", "sadness"),
    ("I felt emotionally drained after the game.", "sadness"),
    ("The themes resonated with my own sadness.", "sadness"),
    ("I was overwhelmed by a sense of loss.", "sadness"),
    ("The game stirred up old wounds.", "sadness"),
    ("I felt a heavy heart throughout.", "sadness"),
    ("The narrative was a poignant reminder of sorrow.", "sadness"),
    ("I was enveloped in a blue mood.", "sadness"),
    ("The game's message was deeply saddening.", "sadness"),
    ("I couldn't help but feel despondent.", "sadness"),
    ("The experience was emotionally taxing.", "sadness"),
    ("I was left with a sense of melancholy.", "sadness"),

    # Social Isolation
    ("I felt disconnected from the game's world.", "social isolation"),
    ("There was a barrier between me and others.", "social isolation"),
    ("I was excluded from every group I approached.", "social isolation"),
    ("The game emphasized my separation from others.", "social isolation"),
    ("I was on the outside looking in.", "social isolation"),
    ("Interactions were superficial and fleeting.", "social isolation"),
    ("I couldn't form any meaningful connections.", "social isolation"),
    ("The game highlighted my social detachment.", "social isolation"),
    ("I was always the observer, never the participant.", "social isolation"),
    ("Attempts to join were met with resistance.", "social isolation"),
    ("I felt like an intruder in every scene.", "social isolation"),
    ("The game portrayed a world where I didn't belong.", "social isolation"),
    ("I was perpetually sidelined.", "social isolation"),
    ("Engagements were brief and unfulfilling.", "social isolation"),
    ("I was isolated despite being among others.", "social isolation"),
    ("The game underscored my social alienation.", "social isolation"),
    ("I was present, but not included.", "social isolation"),
    ("I felt like an outsider throughout.", "social isolation"),
    ("The experience mirrored my feelings of exclusion.", "social isolation"),
    ("I was surrounded, yet entirely alone.", "social isolation"),

    # Unknown
    ("I'm not sure how I feel after playing.", "unknown"),
    ("The game left me with mixed emotions.", "unknown"),
    ("I can't quite put my feelings into words.", "unknown"),
    ("It was an unusual experience; I'm uncertain about it.", "unknown"),
    ("I'm conflicted about the emotions it evoked.", "unknown"),
    ("The game was thought-provoking, but I'm not sure how I feel.", "unknown"),
    ("I need more time to process the experience.", "unknown"),
    ("It's hard to describe the impact it had on me.", "unknown"),
    ("I have a lot to think about after playing.", "unknown"),
    ("The emotions are complex and hard to define.", "unknown"),
    ("I'm left with questions rather than answers.", "unknown"),
    ("The game was enigmatic; my feelings are too.", "unknown"),
    ("I experienced a range of emotions, none clear.", "unknown"),
    ("It's a puzzle, both in gameplay and emotion.", "unknown"),
    ("I can't categorize how it made me feel.", "unknown"),
    ("The experience was ambiguous and open-ended.", "unknown"),
    ("I'm uncertain about the emotional journey it took me on.", "unknown"),
    ("The game defies easy emotional labels.", "unknown"),
    ("I'm still reflecting on what it meant to me.", "unknown"),
    ("The feelings it stirred are indescribable.", "unknown")
]

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "label"])

@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")


model = load_model()
X = model.encode(df["text"].tolist())
y = df["label"]

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X, y)

# Streamlit interface
st.title("🎮 Post-Game Emotion Classifier")
st.write("Enter how you feel after playing the game *Loneliness*.")

user_input = st.text_area("How are you feeling?", height=100)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        user_vector = model.encode([user_input])
        prediction = clf.predict(user_vector)[0]
        st.success(f"🧠 The model thinks this indicates **{prediction.upper()}**.")