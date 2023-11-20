"""
    Libraries
"""
from flask import Flask
from flask import jsonify
from flask_restful import reqparse
from run_nlp import NlpPredict

app = Flask(__name__)

"""
    Parser 
"""
parser = reqparse.RequestParser()
parser.add_argument("komentar", type=str, required=True, help="Komentar harus diisi")
parser.add_argument("rating", type=int, required=True, help="Rating harus diisi")
"""
    Resouce 
"""
@app.errorhandler(500)
def internal_server_error(e):
    return jsonify(error=str(e)), 500

@app.route('/predict', methods=['POST'])
def predict():
    args = parser.parse_args()
    model = NlpPredict('nlp_model/sentimentanalysisv4.h5')
    model.set_sentence(args["komentar"])
    nlp_score = model.predict()
    if nlp_score <= 0.325:
        lebel = "Positif"
    else:
        lebel = "Negatif"

    rating_model_sum = model.sumRatingModel(args["rating"], lebel, nlp_score)

    if rating_model_sum == None:
        rating_model_sum = 0

    return {"data": {
        "nlp_score": nlp_score,
        "rating_model_sum": rating_model_sum
    }}, 200

"""
    Main 
"""
if __name__ == '__main__':
    app.run()