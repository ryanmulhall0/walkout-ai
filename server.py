from flask import Flask, request, jsonify, render_template, redirect, session
import app as walkout  # imports your app.py without starting CLI
import stripe
import os
import psycopg2

flask_app = Flask(__name__)
app = flask_app
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
def init_db():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT UNIQUE,
            google_id TEXT UNIQUE,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            premium_active BOOLEAN DEFAULT FALSE,
            weekly_count INTEGER DEFAULT 0,
            week_start DATE
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

init_db()


# ---------------------------
# Stripe config (test mode)
# ---------------------------
stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PRICE_ID = os.getenv("STRIPE_PRICE_ID", "")
@flask_app.get("/")
def home():
    return render_template("index.html")

@flask_app.post("/ask")
def ask():
    try:
        data = request.get_json(force=True) or {}
        question = (data.get("question") or "").strip()

        if not question:
            return jsonify({"answer": "Type a question."})

        # Call the existing CLI logic from app.py
        answer = walkout.handle_query(question)

        if answer is None:
            return jsonify({"answer": ""})

        if answer == "__QUIT__":
            return jsonify({"answer": "Quit is disabled in web mode."})

        return jsonify({"answer": str(answer)})

    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"})

@flask_app.post("/create-checkout-session")
def create_checkout_session():
    if not stripe.api_key:
        return jsonify({"error": "Stripe secret key is not set on the server."}), 500
    if not STRIPE_PRICE_ID:
        return jsonify({"error": "STRIPE_PRICE_ID is missing. Add it in Render Environment."}), 500

    base = request.host_url.rstrip("/")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"{base}/?paid=1",
            cancel_url=f"{base}/?canceled=1",
        )
        return jsonify({"url": session.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    answer = walkout.handle_query(question)
    if answer is None:
        answer = ""
    if answer == "__QUIT__":
        answer = "Quit command disabled on web."
    return jsonify({"answer": str(answer)})
@app.route("/login/google")
def login_google():
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    redirect_uri = "https://walkout-ai.onrender.com/auth/google/callback"

    return redirect(
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={google_client_id}"
        "&response_type=code"
        "&scope=openid%20email"
        f"&redirect_uri={redirect_uri}"
    )
@app.route("/auth/google/callback")
def google_callback():
    code = request.args.get("code")
    if not code:
        return "Google login failed", 400

    session["user"] = {"email": "google-user"}
    return redirect("/")

if __name__ == "__main__":
    flask_app.run(host="127.0.0.1", port=5000, debug=True)
