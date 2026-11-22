# NERI — Adaptive Learning Roadmap Agent

VS Code version based on the supplied NERI implementation guide.

## Setup
1. Install Python and open this folder in VS Code.
2. Create a virtual environment:
   `python -m venv venv`
3. Windows activation:
   `venv\Scripts\activate`
4. Install packages:
   `pip install -r requirements.txt`
5. Copy `.env.example` to `.env`.
6. Add your Groq API key and supported model name.
7. Run:
   `python app.py`

Choose 2, 4, or 8 weeks. The completed PDF is written to
`generated_roadmaps/`.

This version follows the supplied guide and does not require
Streamlit, FastAPI, a database, or deployment.
