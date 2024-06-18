from flask import Flask, request, jsonify
import os
import json

app = Flask(__name__)



@app.route(os.getenv("ALLOWED_NUMBERPLATES_URL", "/"), methods=['GET'])
def get_numberplates():
    numberplates = json.loads(os.getenv("ALLOWED_NUMBERPLATES_LIST", "[]"))
    return jsonify(numberplates)



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
