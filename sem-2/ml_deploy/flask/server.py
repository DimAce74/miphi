from flask import Flask, request
app = Flask(__name__)

@app.route('/hello')
def hello_func():
    name = request.args.get('name')
    return f'hello {name}!'

@app.route('/')
def index():
    return 'Test message. The server is running'

if __name__ == '__main__':
    app.run(app.run('localhost', 5000, debug=True))

