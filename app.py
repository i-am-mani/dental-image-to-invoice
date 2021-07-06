from image_prediction import predict_image
from flask import Flask, request
from flask import render_template
from collections import Counter
import base64
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

pricing_dict = {'changshu_glasionomer': 2400, 'dpi_impression_paste': 400, 'gc_type_2_mini':1200,
       'gc_type_9': 4500, 'gp_point': 350, 'hybond_cx_smart': 2900, 'kfile': 200, 'lute_glass': 800,
       'orafil_g_plus': 150, 'pearsals_suture_thread': 800, 'prime_chroma_alginate': 450,
       'prime_templute': 120, 'progel_anesthetic': 2500, 'romsons_needle': 150}

def get_invoice(predictions):
    invoice_data = []
    all_products = []
    delivery_total = 0
    for pred in predictions:
        all_products.append(pred['prediction'])

    products_counter = Counter(all_products)

    for product in products_counter:
        product_price = pricing_dict[product]
        product_name = product.replace('_', ' ')
        product_name = product.capitalize()
        qty = products_counter[product]
        invoice_data.append({
            'name': product_name, 'price': product_price, 
            'total': qty*product_price, 'qty': qty
            })
        delivery_total += qty*product_price

    return {'products':invoice_data, 'delivery_total': delivery_total}

@app.route('/', methods=['GET', 'POST'])
def index():
    dict = {}
    if 'sample_image' in request.files:
        img = request.files['sample_image'] 
        path = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        img.save(path)

        if img != None:
            img = cv2.imread(path)
            # img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            encoded_image = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
            predictions = predict_image(path)

            dict = {
                'img': f'data:image/png;base64, {encoded_image}',
                'predictions': predictions,
                'invoice': get_invoice(predictions)
            }


    return render_template('index.html', context=dict)

    
if __name__ == '__main__':
    app.run(debug=True)