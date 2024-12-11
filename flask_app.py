from flask import Flask, Response

app = Flask(__name__)

@app.route('/')
def index():
    with open("w.html", "r") as html_file:
        content = html_file.read()
    return Response(content, mimetype="text/html")

if __name__ == '__main__':
    app.run(debug=True)