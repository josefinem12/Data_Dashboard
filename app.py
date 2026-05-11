#!/Users/josefinemertens/anaconda3/bin/python
from flask import Flask, jsonify, g, send_from_directory
import sqlite3
import json
import os

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), 'tover.db')


def get_db():
    db = getattr(g, '_db', None)
    if db is None:
        db = g._db = sqlite3.connect(DB_PATH)
    return db


@app.teardown_appcontext
def close_db(exc):
    db = getattr(g, '_db', None)
    if db is not None:
        db.close()


@app.route('/')
def index():
    return send_from_directory('.', 'data_dashboard.html')


@app.route('/api/analytics')
def analytics():
    db = get_db()
    rows = db.execute(
        "SELECT key, value FROM analytics WHERE key != 'game_data'"
    ).fetchall()
    data = {row[0]: json.loads(row[1]) for row in rows}
    return jsonify(data)


@app.route('/api/game-data')
def game_data():
    db = get_db()
    row = db.execute(
        "SELECT value FROM analytics WHERE key = 'game_data'"
    ).fetchone()
    if row is None:
        return jsonify({}), 404
    return app.response_class(row[0], mimetype='application/json')


@app.route('/api/hostname-data')
def hostname_data():
    db = get_db()
    row = db.execute(
        "SELECT value FROM analytics WHERE key = 'hostname_data'"
    ).fetchone()
    if row is None:
        return jsonify({}), 404
    return app.response_class(row[0], mimetype='application/json')


if __name__ == '__main__':
    if not os.path.exists(DB_PATH):
        print("ERROR: tover.db not found. Run compute_analytics.py first.")
    else:
        print("Starting dashboard at http://localhost:5000")
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
