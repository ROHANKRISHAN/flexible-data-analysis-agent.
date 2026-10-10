import json, os
from analytics import answer_question, profile_dataset, summarize_numeric, grouped_analysis, time_analysis

SYSTEM = """You are a data analysis agent. You will receive a dataset schema and a user question.
Select only columns that exist in the provided schema. Never invent column names or values.
Choose one analysis plan: summary, group, or time. Return a JSON object with keys kind, metric,
dimension, date, aggregation, and explanation. Use aggregation sum, mean, median, min, max, or count.
If the question cannot be answered from available columns, state that in explanation. Do not claim causation from descriptive analysis."""

def run_agent(df, question):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return answer_question(df, question)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        schema = profile_dataset(df)
        prompt = {"question":question,"schema":schema}
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),
            instructions=SYSTEM + "\nReturn JSON only.",
            input=json.dumps(prompt, default=str),
            text={"format":{"type":"json_object"}}
        )
        plan = json.loads(response.output_text)
        cols = set(df.columns)
        kind = plan.get("kind")
        metric = plan.get("metric")
        if metric not in cols:
            raise ValueError("Model selected a metric that is not present in the dataset.")
        agg = plan.get("aggregation","sum")
        if kind == "group":
            dimension = plan.get("dimension")
            if dimension not in cols:
                raise ValueError("Model selected a dimension that is not present in the dataset.")
            data = grouped_analysis(df, dimension, metric, agg)
            return {"summary":plan.get("explanation","Grouped analysis completed."),"data":data,"plan":plan,"mode":"llm"}
        if kind == "time":
            date_col = plan.get("date")
            if date_col not in cols:
                raise ValueError("Model selected a date column that is not present in the dataset.")
            data = time_analysis(df,date_col,metric,agg if agg in {"sum","mean","count"} else "sum")
            return {"summary":plan.get("explanation","Time analysis completed."),"data":data,"plan":plan,"mode":"llm"}
        return {"summary":plan.get("explanation","Summary completed."),"data":summarize_numeric(df,metric),"plan":plan,"mode":"llm"}
    except Exception as exc:
        result = answer_question(df,question)
        result["mode"] = "offline_fallback"
        result["note"] = f"LLM planning was unavailable or invalid ({type(exc).__name__}); used schema-aware offline analysis."
        return result
