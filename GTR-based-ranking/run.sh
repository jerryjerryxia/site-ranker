#!/bin/bash
# Run the DACIS dashboard

cd "$(dirname "$0")/.."
source venv/bin/activate
cd GTR-based-ranking
streamlit run app.py --server.headless true "$@"
