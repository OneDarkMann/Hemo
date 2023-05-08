from flask import Flask,request,render_template,redirect
import os
from model.model import pipline

app = Flask(__name__)

dir = os.getcwd()
print(dir)

app.config["IMAGE_UPLOADS"] = str(dir)"/static/Images/inputs"
app.config["IMAGE_DOWNLOADS"] = str(dir)"/static/Images/outputs"
#app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["PNG","JPG","JPEG"]

from werkzeug.utils import secure_filename


@app.route('/',methods = ["GET","POST"])
def upload_image():
	if request.method == "POST":
		image = request.files['file']

		if image.filename == '':
			print("Image must have a file name")
			return redirect(request.url)


		filename = secure_filename(image.filename)

		basedir = os.path.abspath(os.path.dirname(__file__))
		image.save(os.path.join(basedir,app.config["IMAGE_UPLOADS"],filename))
		
        # put model to return predection of the model in the shape of image
		filename_out, name_out = pipline(img_path=app.config["IMAGE_UPLOADS"]+"/"+filename)

		filename_out.save(os.path.join(basedir,app.config["IMAGE_DOWNLOADS"],filename))
        # and also do the same step of the saving file path to the prediction folder

		return render_template("index.html",filename=filename, filename_out = name_out)



	return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='/Images/inputs/'+filename), code=301)


app.run(debug=True,port=2000)
