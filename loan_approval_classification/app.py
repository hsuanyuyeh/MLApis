from fastapi import FastAPI
import pickle
import pandas as pd
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

model_filename = 'model.pkl'
transformer_filename = 'transformer.pkl'
model = pickle.load(open(model_filename, 'rb'))
transformer = pickle.load(open(transformer_filename, 'rb'))
cat_col = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
num_col = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']


app = FastAPI()

class mlapi_response:
    def __init__(self, model=None, transformer=None, cat_col=None, num_col=None):
        self.model = model
        self.input = None
        self.transformed_input = None
        self.result = None
        self.cat_col = cat_col
        self.num_col = num_col
        self.transformer = transformer

    def read_input(self, input: pd.DataFrame):
        self.input = input
        self.transformed_input = self.transformer.transform(self.input[self.cat_col+self.num_col])

    def predict_result(self):
        result = self.model.predict(self.transformed_input)
        self.result = "Approved" if result[0] else "Rejected"
    
mlmodel = mlapi_response(model, transformer, cat_col, num_col)
@app.get("/input/")
def receive_data(input):
    df = pd.read_json(input)
    mlmodel.read_input(df)
    mlmodel.predict_result()

@app.post("/predict/")
def predict_data():
    return mlmodel.result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    model_filename = 'model.pkl'
    transformer_filename = 'transformer.pkl'
    model = pickle.load(open(model_filename, 'rb'))
    transformer = pickle.load(open(transformer_filename, 'rb'))
    cat_col = ['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
    num_col = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
    
    mlmodel = mlapi_response(model, transformer, cat_col=cat_col, num_col=num_col)
    xtest = pd.read_csv('loan_data.csv').head(1)[cat_col+num_col]

    mlmodel.read_input(xtest)
    mlmodel.predict_result()
    print(f'result: {mlmodel.result}')