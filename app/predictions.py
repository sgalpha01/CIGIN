api_names = ["/predict",'/predict_solubility','/predict_two','/predict_solubility_json']
from app.worker import predictions, predictions_two

predict_dum = {
    0: predictions,
    1: predictions_two
}