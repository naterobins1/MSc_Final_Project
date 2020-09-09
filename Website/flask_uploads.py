from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/upload')
def upload_file_temp():
    return render_template('dad_webpage.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))

    return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)

#http://localhost:5000/upload
#this webpage allows me to upload a file, so now I just need to be able to do stuff with the files I think