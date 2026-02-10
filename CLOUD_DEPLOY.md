# Cloud Deployment Guide (Free Tier)

This guide explains how to deploy your Voice Assistant to the cloud so it runs 24/7, even when your computer is off.
We will use **Render.com** (App hosting) and **MongoDB Atlas** (Database). Both have generous free tiers.

## Prerequisites
- [GitHub Account](https://github.com/)
- [Render Account](https://render.com/)
- [MongoDB Atlas Account](https://www.mongodb.com/atlas/database)
- **Groq API Key** (Essential for cloud deployment as it replaces heavy local AI models)

---

## Step 1: Set up the Database (MongoDB Atlas)
Since Render's free tier doesn't include a persistent database, we use Atlas.

1.  Log in to [MongoDB Atlas](https://www.mongodb.com/atlas/database).
2.  Create a new Project (e.g., "VoiceAssist").
3.  **Build a Database**: Select **M0 (Free)** tier.
4.  **Security Quickstart**:
    -   Create a database user (username/password). **Remember this!**
    -   Add IP Address: `0.0.0.0/0` (Allow access from anywhere/Render).
5.  **Connect**:
    -   Click "Connect" -> "Drivers".
    -   Copy the connection string. It looks like:
        `mongodb+srv://<username>:<password>@cluster0.abcde.mongodb.net/?retryWrites=true&w=majority`
    -   Replace `<username>` and `<password>` with your actual credentials.

---

## Step 2: Prepare Your Code
1.  Push your code to a new **GitHub Repository**.
    -   Ensure `requirements.txt` is in the root.
    -   Ensure `backend.py` is in the root.

---

## Step 3: Deploy to Render
1.  Log in to [Render Dashboard](https://dashboard.render.com/).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  **Configure the Service**:
    -   **Name**: `voice-assist` (or similar)
    -   **Region**: Closest to you (e.g., Singapore/Ohio).
    -   **Branch**: `main`
    -   **Runtime**: **Python 3**
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
    -   **Instance Type**: **Free**

5.  **Environment Variables** (The most important part!):
    Add these key-value pairs:
    
    | Key | Value |
    | :--- | :--- |
    | `PYTHON_VERSION` | `3.10.0` |
    | `LLM_PROVIDER` | `groq` |
    | `STT_PROVIDER` | `groq` |
    | `GROQ_API_KEY` | `your_actual_groq_api_key` |
    | `GROQ_MODEL` | `llama3-70b-8192` |
    | `MONGO_URI` | `your_mongodb_atlas_connection_string` |
    | `MONGO_DB_NAME` | `voice_assist_db` |

    *Note: Setting `STT_PROVIDER=groq` is crucial. It tells the app to use Groq's cloud API for speech recognition instead of trying to load heavy models on the small free server.*

6.  Click **Create Web Service**.

---

## Step 4: Verification
1.  Wait for the build to finish (might take 2-3 minutes).
2.  Once deployed, Render gives you a URL (e.g., `https://voice-assist-xyz.onrender.com`).
3.  Open that URL in your browser.
4.  **Microphone Permission**: Since Render provides HTTPS automatically, microphone permissions will work perfectly!

## Troubleshooting
-   **"Application Error"**: Check the "Logs" tab in Render.
-   **Database Error**: Double-check your `MONGO_URI` and ensure you allowed `0.0.0.0/0` in Atlas Network Access.
-   **Slow Response**: The free tier on Render "spins down" after inactivity. The first request might take 50 seconds. Subsequent requests will be fast.
