# How to Share & Deploy Your Voice Assistant

You have two main ways to share your assistant:
1.  **Temporary Sharing (Ngrok)**: Best for quick demos to friends. Runs on your computer.
2.  **Permanent Deployment (Render)**: Best for 24/7 access. Runs on the cloud.

---

## Option 1: Share Instantly with Friends (Ngrok)
This method makes your local server accessible via a public URL for as long as your computer is on.

### 1. Install Ngrok
- Go to [ngrok.com](https://ngrok.com/download) and sign up (it's free).
- Download and unzip ngrok.
- Connect your account:
  ```powershell
  ngrok config add-authtoken <YOUR_AUTH_TOKEN>
  ```

### 2. Start Your App
Open a terminal in your project folder and run:
```powershell
run.bat
```
Ensure it's running on `http://localhost:8000`.

### 3. Start the Tunnel
**Important**: If you just installed ngrok, **restart VS Code** completely so it detects the new command.

Alternatively, use the included helper script:
```powershell
share.bat
```

Or manually run:
```powershell
ngrok http 8000
```

### 4. Share the Link
- Copy the **Forwarding** URL from the ngrok terminal (e.g., `https://a1b2-c3d4.ngrok-free.app`).
- Send this link to your friends.
- **Note**: They must use `https` (not http) to allow microphone access in their browser.

---

## Option 2: Deploy to Cloud (Render.com)
This hosts your app permanently for free/cheap.

### 1. Prepare for Cloud
Ensure your `requirements.txt` is up to date. (It already looks good!).
Ensure your project is pushed to a GitHub repository.

### 2. Create Render Service
1.  Log in to [dashboard.render.com](https://dashboard.render.com/).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  **Settings**:
    -   **Runtime**: Python 3
    -   **Build Command**: `pip install -r requirements.txt`
    -   **Start Command**: `uvicorn backend:app --host 0.0.0.0 --port $PORT`
5.  **Environment Variables**:
    Add your API keys under the "Environment" tab:
    -   `GROQ_API_KEY`: `...`
    -   `GROQ_MODEL`: `llama3-70b-8192`
    -   `LLM_PROVIDER`: `groq`
6.  Click **Create Web Service**.

### 3. Share
Render will give you a URL (e.g., `https://voice-assist.onrender.com`). Share this with anyone!

---

## Important Note on Permissions
Browsers block microphone access on insecure connections (`http://`).
- **Localhost** is an exception (works on http).
- **Ngrok** provides HTTPS automatically (use the `https://` link).
- **Render** provides HTTPS automatically.

Always share the **HTTPS** link.
