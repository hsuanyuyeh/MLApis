## Loan application status prediction app\
Build a loan application status prediction app with lightgbm, fastapi and streamlit.\
\
**Steps**\
1. Explore the loan data with data profiling\
2. Build and save the lightGBM model in jupyter notebook script_model.ipynb\
3. Create a ml model backend server with fastapi loading the saved model\
4. Build the front end ui with streamlit. Customize the input data in sidebar\
\
Run the backend server: uvicorn server:app --host 0.0.0.0 --port 8000\
Run the chatbot ui: streamlit run client.py

