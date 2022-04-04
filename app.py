from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
app = Flask(__name__)


from simpletransformers.t5 import T5Model , T5Args

#python -m pip install -r requirements.txt
model_args = T5Args()
model_args.num_train_epochs = 3
#model_args.no_save = True
#model_args.evaluate_generated_text = True
#model_args.evaluate_during_training = True
#model_args.evaluate_during_training_verbose = True
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.use_cuda = False
model_args.use_multiprocessing = False
model_args.use_multiprocessing_for_evaluation = False
model_args.use_multiprocessed_decoding = False
model_args.learning_rate=0.001
#model_args.num_beams = 3
model_args.train_batch_size = 4
model_args.eval_batch_size = 4
model_args.adafactor_beta1 = 0
model_args.length_penalty=1.5
model_args.max_length=100
model_args.max_seq_length = 100


model = T5Model("mt5", "google/mt5-base", args=model_args , use_cuda=False)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    print('Request for predict page received')
    
    text = request.json.get('text')
    print(text)
    if text:
        
        p = model.predict([text])[0]
        print(f'Request for predict for {text} is {p}')
        
        #return render_template('predict.html', text = p)
        return p
    else:
        print('Request for predict page received with no text or blank text -- redirecting')
        return None



if __name__ == '__main__':
   app.run()