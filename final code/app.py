from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
import sqlite3
import joblib
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPool2D,GlobalAveragePooling2D

json_file1 = open('models/model1.json', 'r')
loaded_model_json1 = json_file1.read()
json_file1.close()
model1 = model_from_json(loaded_model_json1)
model1.load_weights("models/skin_model.hdf5")

json_file2 = open('models/model2.json', 'r')
loaded_model_json2 = json_file2.read()
json_file2.close()
model2 = model_from_json(loaded_model_json2)
model2.load_weights("models/skin_model2.hdf5")

json_file3 = open('models/model3.json', 'r')
loaded_model_json3 = json_file3.read()
json_file3.close()
model3 = model_from_json(loaded_model_json3)
model3.load_weights("models/model.hdf5")


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


label_mapping1 = {
    0: "Atopic Dermatitis",
    1: 'Eczema'
}
dis1={"Atopic Dermatitis":"Atopic dermatitis: Atopic dermatitis (AD), also known as atopic eczema, is a long-term type of inflammation of the skin (dermatitis). It results in itchy, red, swollen, and cracked skin.Scratching the affected areas worsens the symptoms. Many people with atopic dermatitis develop hay fever or asthma.The cause is believed to involve genetics, immune system dysfunction, environmental exposures, and difficulties with the permeability of the skin.Moisturizing regularly and following other skin care habits can relieve itching and prevent new outbreaks (flares).",
      "Eczema":"Eczema: Eczema is a common skin condition that causes itchy, red, dry, and irritated skin.Eczema tends to flare when your skin is exposed to external irritants, which cause your immune system to overreact. Eczema can occur anywhere but usually affects the arms, inner elbows, backs of the knees, cheeks, and scalp. It’s not contagious. Moisturize your skin regularly or when your skin becomes dry.Take baths or showers with warm, not hot, water and stay hydrated."}


label_mapping2 = {
    0: "Basal Cell Carcinoma",
    1: 'Melanocytic Nevi',
    2: 'Melanoma'
}
dis2={"Basal Cell Carcinoma":"Basal Cell Carcinoma: Basal cell carcinoma is a type of skin cancer. Basal cell carcinoma begins in the basal cells — a type of cell within the skin that produces new skin cells as old ones die off. Basal cell carcinoma often appears as a slightly transparent bump on the skin, though it can take other forms.Most basal cell carcinomas are thought to be caused by long-term exposure to ultraviolet (UV) radiation from sunlight. Avoiding the sun and using sunscreen may help protect against basal cell carcinoma.",
      "Melanocytic Nevi":"Melanocytic Nevi: A usually non-cancerous disorder of pigment-producing skin cells commonly called birth marks or moles.This type of mole is often large and caused by a disorder involving melanocytes, cells that produce pigment (melanin).Melanocytic nevi can be rough, flat or raised. They can exist at birth or appear later. Rarely, melanocytic nevi can become cancerous.Most cases don't require treatment, but some cases require removal of the mole.The number of melanocytic nevi can be minimized by strict protection from the sun.",
      "Melanoma":"Melanoma: Melanoma is the most invasive skin cancer with the highest risk of death. Prevention and early treatment are critical.Melanoma comes from skin cells called melanocytes. These cells produce melanin, the dark pigment that gives skin its color. Most melanomas are black or brown in color, but some are pink, red, purple or skin-colored.Most experts agree that a major risk factor for melanoma is overexposure to sunlight, especially sunburns when you are young.Avoid sun and seek shade, especially between 10 a.m. and 4 p.m. Early detection is important to minimize the risks associated with melanoma."}
label_mapping3 = {
    0: "Alopecia Areata",
    1: 'Folliculitis',
    2: 'Psoriasis',
    3: 'Seborrheic Dermatitis',
    4: 'Tinea Capitis'
}
dis3={"Alopecia Areata":"Alopecia Areata: Alopecia areata is a disease that happens when the immune system attacks hair follicles and causes hair loss.Hair typically falls out in small, round patches about the size of a quarter, but in some cases, hair loss is more extensive. Most people with the disease are healthy and have no other symptoms.In cases of relatively mild alopecia areata,hair may regrow without treatment.Traditional treatments for alopecia areata include steroid injections to the areas where the hair has been shed and topical and oral medications.",
      "Folliculitis":"Folliculitis: Folliculitis is a common skin condition that happens when hair follicles become inflamed. It's often caused by an infection with bacteria. At first it may look like small pimples around the hair follicles. The condition can be itchy, sore and embarrassing. The infection can spread and turn into crusty sores.Use antibacterial cleansers to clean the skin. This will limit the amount of bacteria on the skin.",
      "Psoriasis":"Psoriasis: Scalp psoriasis is a common skin disorder. It may look different on different skin tones. For light- to medium-skinned, it often shows up as raised, reddish or salmon-colored patches with white scales. On darker skin, the patches may be purple and the scales gray. It can be a single patch or several, and can even affect the entire scalp. It can also spread to the forehead, the back of the neck, or behind and inside the ears.The best way to manage scalp psoriasis is to apply medication according to the instructions of a healthcare professional which includes medicated shampoos and scalp softeners,if the psoriasis on the scalp is thick.",
      "Seborrheic Dermatitis":"Seborrheic Dermatitis: Seborrheic dermatitis is a common skin condition that mainly affects the scalp.It causes scaly patches, inflamed skin and stubborn dandruff.This condition can be irritating but it's not contagious, and it doesn't cause permanent hair loss.Seborrheic dermatitis may go away without treatment. Or you may need to use medicated shampoo or other products long term to clear up symptoms and prevent flare-ups. Seborrheic dermatitis is also called dandruff, seborrheic eczema and seborrheic psoriasis.",
      "Tinea Capitis":"Tinea Capitis: Tinea capitis, also known as ringworm or herpes tonsurans infection, is a fungal infection of the scalp hair.The fungi can penetrate the hair follicle's outer root sheath and ultimately may invade the hair shaft.Mold-like fungi called dermatophytes cause tinea capitis. Tinea capitis also spreads very easily.Fungi thrive in warm, moist environments. It commonly grows in tropical places.Avoid sharing personal items such as hats, hairbrushes, combs, pillows and helmets.Keep your scalp clean and dry.Wash pillows, sheets and other bedding frequently.Wash your hands after petting, playing with or coming into contact with pets."}

with sqlite3.connect('user.db') as db:
    c = db.cursor()

c.execute('CREATE TABLE IF NOT EXISTS user (id INTEGER PRIMARY KEY AUTOINCREMENT,firstname TEXT NOT NULL,lastname TEXT NOT NULL,email TEXT NOT NULL,phone TEXT NOT NULL,gender TEXT NOT NULL,age INTEGER NOT NULL,result TEXT NULL);')
db.commit()
db.close()

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/model1')
def model11():
    return render_template('model1.html',msg="none")


@app.route('/model2')
def model12():
    return render_template('model2.html',msg="none")

@app.route('/model3')
def model13():
    return render_template('model3.html',msg="none")




@app.route('/result1', methods=['GET','POST'])
def result1():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resized_image = np.asarray(Image.open('static/uploads/'+filename).resize((224,224)))
            image_array = np.asarray(resized_image.tolist())
            test_image = image_array.reshape(1,224,224,3)
            prediction_class = model1.predict(test_image)
            print(prediction_class)
            prediction_class = np.argmax(prediction_class,axis=1)
            pred=label_mapping1[prediction_class[0]]
            dis=dis1[pred]
            conn = sqlite3.connect('user.db')
            c = conn.cursor()
            c.execute('INSERT INTO user (firstname, lastname,email,phone,gender,age,result) VALUES (?, ?, ?, ?, ?, ?, ?)', (firstname, lastname,email,phone,gender,age,pred))
            conn.commit()
            return render_template('result1.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender,d=dis)

        else:
            return render_template('model1.html',msg='Allowed image types are - png, jpg, jpeg')





@app.route('/result2', methods=['POST'])
def result2():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resized_image = np.asarray(Image.open('static/uploads/'+filename).resize((224,224)))
            image_array = np.asarray(resized_image.tolist())
            test_image = image_array.reshape(1,224,224,3)
            prediction_class = model2.predict(test_image)
            prediction_class = np.argmax(prediction_class,axis=1)
            pred=label_mapping2[prediction_class[0]]
            dis=dis2[pred]
            conn = sqlite3.connect('user.db')
            c = conn.cursor()
            c.execute('INSERT INTO user (firstname, lastname,email,phone,gender,age,result) VALUES (?, ?, ?, ?, ?, ?, ?)', (firstname, lastname,email,phone,gender,age,pred))
            conn.commit()
            return render_template('result2.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender,d=dis)
            
        else:
            return render_template('model2.html',msg='Allowed image types are - png, jpg, jpeg')


@app.route('/result3', methods=['POST'])
def result3():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            resized_image = np.asarray(Image.open('static/uploads/'+filename).resize((224,224)))
            image_array = np.asarray(resized_image.tolist())
            test_image = image_array.reshape(1,224,224,3)
            prediction_class = model3.predict(test_image)
            prediction_class = np.argmax(prediction_class,axis=1)
            pred=label_mapping3[prediction_class[0]]
            dis=dis3[pred]
            conn = sqlite3.connect('user.db')
            c = conn.cursor()
            c.execute('INSERT INTO user (firstname, lastname,email,phone,gender,age,result) VALUES (?, ?, ?, ?, ?, ?, ?)', (firstname, lastname,email,phone,gender,age,pred))
            conn.commit()
            return render_template('result3.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender,d=dis)
            
        else:
            return render_template('model3.html',msg='Allowed image types are - png, jpg, jpeg')



@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
