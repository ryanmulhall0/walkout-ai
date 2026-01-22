from flask import Flask, request, jsonify, render_template, redirect, session
import app as walkout  # imports your app.py without starting CLI
import stripe
import os
import psycopg2
import requests
import uuid
from urllib.parse import urlencode
from datetime import date
flask_app = Flask(__name__)
app = flask_app
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret")
def init_db():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")

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

    cur.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id SERIAL PRIMARY KEY,
            email TEXT NOT NULL,
            asked_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    used = None
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

        return jsonify({"answer": answer, "used": used, "limit": 5})


    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"})

@flask_app.post("/create-checkout-session")
@app.post("/stripe/create-checkout")
def create_checkout_session():
    # Must be logged in
    u = session.get("user")
    if not u or not u.get("email"):
        return jsonify({"error": "Not logged in"}), 401
    # ===== FREE LIMIT: 5 questions per rolling 7 days (anonymous OK) =====
    # Identify the user (logged in => email, anonymous => anon:<uuid>)
    email = ""
    if u and u.get("email"):
        email = (u.get("email") or "").strip().lower()
        ident = email
        logged_in = True
    else:
        anon_id = session.get("anon_id")
        if not anon_id:
            anon_id = uuid.uuid4().hex
            session["anon_id"] = anon_id
        ident = f"anon:{anon_id}"
        logged_in = False

    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    # Premium only applies to logged-in users (anonymous is always free)
    is_premium = False
    if logged_in:
        cur.execute("SELECT premium_active FROM users WHERE email=%s;", (ident,))
        row = cur.fetchone()
        is_premium = bool(row and row[0])

    if not is_premium:
        # Count questions in last 7 days for this ident
        cur.execute("""
            SELECT COUNT(*) FROM questions
            WHERE email=%s AND asked_at >= NOW() - INTERVAL '7 days';
        """, (ident,))
        used = int(cur.fetchone()[0] or 0)

        if used >= 5:
            cur.close()
            conn.close()
            return jsonify({
                "answer": "Free limit reached: 5 questions per week. Please upgrade to Premium for unlimited access.",
                "limit_reached": True,
                "used": used,
                "limit": 5,
                "logged_in": logged_in,
                "premium": False
            }), 429

        # Log this question (free)
        cur.execute("INSERT INTO questions (email) VALUES (%s);", (ident,))
        conn.commit()

        # New count after insert
        cur.execute("""
            SELECT COUNT(*) FROM questions
            WHERE email=%s AND asked_at >= NOW() - INTERVAL '7 days';
        """, (ident,))
        used = int(cur.fetchone()[0] or 0)

        cur.close()
        conn.close()
        # Leave `used` available for the response
    else:
        # Premium: log optionally (doesn't matter), but still useful for analytics
        cur.execute("INSERT INTO questions (email) VALUES (%s);", (ident,))
        conn.commit()
        cur.close()
        conn.close()
        used = None
    # ===== END FREE LIMIT =====


    email = u["email"].strip().lower()

    # Check DB for premium
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT premium_active FROM users WHERE email=%s;", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row and row[0]:
        return jsonify({"error": "Already premium"}), 400

    if not stripe.api_key:
        return jsonify({"error": "Stripe secret key is not set on the server."}), 500
    if not STRIPE_PRICE_ID:
        return jsonify({"error": "STRIPE_PRICE_ID is missing. Add it in Render Environment."}), 500

    base = request.host_url.rstrip("/")

    try:
        checkout = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=email,
            line_items=[{"price": STRIPE_PRICE_ID, "quantity": 1}],
            success_url=f"{base}/?paid=1",
            cancel_url=f"{base}/?canceled=1",
        )
        return jsonify({"url": checkout.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/stripe/portal")
def stripe_portal():
    # Must be logged in
    u = session.get("user")
    if not u or not u.get("email"):
        return jsonify({"error": "Not logged in"}), 401

    email = u["email"].strip().lower()

    # Must be premium to manage
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT premium_active FROM users WHERE email=%s;", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row[0]:
        return jsonify({"error": "Not premium"}), 403

    if not stripe.api_key:
        return jsonify({"error": "Stripe secret key is not set on the server."}), 500

    base = request.host_url.rstrip("/")

    try:
        customers = stripe.Customer.list(email=email, limit=1)
        if not customers.data:
            return jsonify({"error": "No Stripe customer found for this email yet."}), 400

        cust_id = customers.data[0].id

        portal = stripe.billing_portal.Session.create(
            customer=cust_id,
            return_url=f"{base}/",
        )
        return jsonify({"url": portal.url})
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

    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    redirect_uri = "https://walkout-ai.onrender.com/auth/google/callback"

    token_resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=10,
    )
    if token_resp.status_code != 200:
        return "Google token exchange failed", 400

    token_data = token_resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        return "Google token missing", 400

    userinfo_resp = requests.get(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    if userinfo_resp.status_code != 200:
        return "Google userinfo failed", 400

    userinfo = userinfo_resp.json()
    email = (userinfo.get("email") or "").strip().lower()
    google_id = (userinfo.get("id") or "").strip()

    if not email:
        return "Google did not return an email", 400

    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    cur.execute("SELECT premium_active FROM users WHERE email=%s;", (email,))
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO users (email, google_id, premium_active, weekly_count, week_start) VALUES (%s, %s, FALSE, 0, %s);",
            (email, google_id or None, date.today()),
        )
        conn.commit()
        premium_active = False
    else:
        premium_active = bool(row[0])

    cur.close()
    conn.close()

    session["user"] = {"email": email, "premium": premium_active}
    return redirect("/")

@app.route("/whoami")
def whoami():
    u = session.get("user")
    if not u:
        return jsonify({"logged_in": False}), 200
    return jsonify({"logged_in": True, "email": u.get("email"), "premium": bool(u.get("premium"))}), 200

@app.get("/premium/status")
def premium_status():
    u = session.get("user")
    if not u or not u.get("email"):
        return jsonify({"logged_in": False, "premium": False}), 200

    email = u["email"].strip().lower()
    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT premium_active FROM users WHERE email=%s;", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    premium = bool(row[0]) if row else False
    session["user"]["premium"] = premium  # keep session in sync
    return jsonify({"logged_in": True, "premium": premium}), 200


if __name__ == "__main__":
    flask_app.run(host="127.0.0.1", port=5000, debug=True)
@app.post("/stripe/webhook")
def stripe_webhook():
    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")

    if not webhook_secret:
        return "Webhook secret not set", 500

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig_header,
            secret=webhook_secret,
        )
    except Exception as e:
        return f"Webhook error: {str(e)}", 400

    if event["type"] == "checkout.session.completed":
        session_obj = event["data"]["object"]

        # Try multiple places Stripe may store the email
        email = None
        if isinstance(session_obj.get("customer_details"), dict):
            email = session_obj["customer_details"].get("email")
        if not email:
            email = session_obj.get("customer_email")
        if not email:
            email = session_obj.get("metadata", {}).get("email")

        if email:
            email = email.strip().lower()
            db_url = os.environ.get("DATABASE_URL")
            conn = psycopg2.connect(db_url)
            cur = conn.cursor()
            cur.execute("UPDATE users SET premium_active=TRUE WHERE email=%s;", (email,))
            conn.commit()
            cur.close()
            conn.close()

    return "ok", 200
