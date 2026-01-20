from flask import Flask, request, jsonify, render_template
import app as walkout  # imports your app.py without starting CLI
import stripe
import os
flask_app = Flask(__name__)
app = flask_app
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
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Type a question."})
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

if __name__ == "__main__":
    flask_app.run(host="127.0.0.1", port=5000, debug=True)
