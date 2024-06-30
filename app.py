from flask import Flask, render_template

app = Flask(__name__)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data_latih')
def data_latih():
    return render_template('data_latih.html')

@app.route('/data_uji')
def data_uji():
    return render_template('data_uji.html')

@app.route('/hasil')
def hasil():
    return render_template('hasil.html')

if __name__ == '__main__':
    app.run(debug=True)
