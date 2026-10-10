# Flexible Data Analysis Agent

A CSV analysis application that does not require fixed business column names. It profiles the uploaded schema, detects likely numeric/date/categorical/identifier fields, offers manual column selection, and supports an optional LLM planner.

## Run on Windows
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## LLM mode (optional)
Copy `.env.example` to `.env`, add your API key, and restart Streamlit. Keep `.env` private. The app works without an API key using schema-aware offline heuristics.

## What is dynamic?
- No required column names such as `Sales`, `Profit`, `Order ID`, or `Order Date`.
- Column roles are inferred from values and broad name hints.
- The LLM receives the uploaded schema and must select existing column names.
- The application validates selected columns before calculating.
- Users can select arbitrary detected columns manually in Explore.

## Limitations
"Any column names" does not mean every question is answerable automatically. A metric needs to be measurable, date questions need a date-like field, and semantic meaning may be ambiguous. Offline mode is heuristic; LLM mode improves interpretation but should still be checked. The demo dataset is synthetic.
