from flask import Flask, request, jsonify, render_template
import app as walkout  # imports your app.py without starting CLI

flask_app = Flask(__name__)

@flask_app.get("/")
def home():
    return render_template("index.html")

@flask_app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"answer": "Type a question."})

    answer = walkout.handle_query(question)
    if answer is None:
        answer = ""
    if answer == "__QUIT__":
        answer = "Quit command disabled on web."
    return jsonify({"answer": str(answer)})

if __name__ == "__main__":
    flask_app.run(host="127.0.0.1", port=5000, debug=True)
