from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

application = Flask(__name__)
app = application


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
   if request.method == 'GET':
      return render_template('home.html')
   else:

      #get the text input
      article = request.form['text']

      #get data as df
      custom_data = CustomData(article)
      data_as_df = custom_data.get_data_as_dataFrame()

      #predict the output
      predict_pipeline = PredictPipeline()
      prediction = predict_pipeline.predict(data_as_df)

      if prediction[0] == 0:
         pred = "Fake News"
      else:
         pred = "Real News"

      return render_template('home.html', prediction_text = f"predicted to be: {pred}")
   
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
