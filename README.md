echo "# Airtel Loan Risk App

This Streamlit app classifies mobile money users into credit risk categories (low, medium, high) based on their age, transaction history, loan experience, and CRB status. It also assigns a proposed loan limit.

## Files

- \`app.py\`: Main Streamlit app
- \`loan_risk_classifier.pkl\`: Trained Random Forest model
- \`risk_label_encoder.pkl\`: Label encoder for risk class
- \`requirements.txt\`: Dependencies for deployment

## How to Run Locally

\`\`\`bash
pip install -r requirements.txt
streamlit run app.py
\`\`\`
" > README.md

git add README.md
git commit -m "Add README.md"
git push
