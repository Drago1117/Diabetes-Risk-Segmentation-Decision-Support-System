import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table
import json
import os
from utils.predict import predict_risk
from utils.cluster import get_cluster

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Diabetes Risk Segmentation Dashboard - BC Analytics", style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    html.Div([
        html.H2("Diabetes Risk Assessment Dashboard", style={'color': '#007bff'}),
        html.P("AI-powered tool for predicting diabetes risk stage and patient segmentation using XGBoost and KMeans. Enter patient data to get personalized risk assessment.", style={'fontSize': '18px', 'color': '#666'})
    ], className="p-5 bg-primary text-white rounded mb-5", style={'textAlign': 'center'}),

    dcc.Tabs([
        dcc.Tab(label='Patient Risk Prediction', children=[
            html.Div([
                html.H3("Patient Input Form", style={'textAlign': 'center', 'marginBottom': '30px'}),
                html.Div([
                    html.Label("Age:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(id='age', type='number', placeholder="Enter age (years)", style={'width': '100%'})
                ], style={'marginBottom': '20px', 'width': '100%'}),
                html.Div([
                    html.Label("Gender:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(id='gender', options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}], style={'width': '100%'})
                ], style={'marginBottom': '20px', 'width': '100%'}),
                html.Div([
                    html.Label("BMI:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Input(id='bmi', type='number', placeholder="Enter BMI", step="0.1", style={'width': '100%'})
                ], style={'marginBottom': '20px', 'width': '100%'}),
                html.Div([
                    html.Label("Physical Activity:", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Dropdown(id='physical_activity', options=[{'label': 'High', 'value': 'High'}, {'label': 'Medium', 'value': 'Medium'}, {'label': 'Low', 'value': 'Low'}], style={'width': '100%'})
                ], style={'marginBottom': '20px', 'width': '100%'}),
                html.Div([
                    html.Label("Diet Score (0-10):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Slider(id='diet_score', min=0, max=10, step=0.5, marks={i: str(i) for i in [0,2,4,6,8,10]})
                ], style={'marginBottom': '20px', 'width': '100%'}),
                html.Div([
                    html.Label("Sleep Hours per Day (0-12):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
                    dcc.Slider(id='sleep_hours_per_day', min=0, max=12, step=0.5, marks={i: str(i) for i in [0,4,8,12]})
                ], style={'marginBottom': '20px', 'width': '100%'}),
                html.Div(html.Button('🔮 Predict Risk & Segment', id='predict-btn', n_clicks=0, className="btn btn-primary btn-lg"), style={'alignSelf': 'center', 'marginBottom': '20px'}),
                html.Div(id='prediction-output')
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'maxWidth': '500px', 'margin': 'auto'})
        ]),
        dcc.Tab(label='Model Insights', children=[
            html.Div([
                html.Div([
                    html.H3("📊 XGBoost Risk Classifier"),
                    html.P("Accuracy: 91% | Leakage-free | Deployment-ready"),
                    html.Ul([
                        html.Li("Targets: No, Pre, Type 1/2, Gestational"),
                        html.Li("Key features: BMI, lifestyle, family history")
                    ], style={'fontSize': '16px'})
                ], className="card p-4 m-3", style={'backgroundColor': '#d4edda'}),
                html.Div([
                    html.H3("🎯 KMeans Clustering"),
                    html.P("3 lifestyle/health segments"),
                    html.Ul([
                        html.Li("Cluster 0: Low Risk - Healthy"),
                        html.Li("Cluster 1: Medium Risk - Monitor"),
                        html.Li("Cluster 2: High Risk - Intervene")
                    ], style={'fontSize': '16px'})
                ], className="card p-4 m-3", style={'backgroundColor': '#d1ecf1'}),
                html.Div([
                    html.H3("💡 Actionable Recommendations"),
                    html.Ul([
                        html.Li("Cluster 2: +30min daily activity"),
                        html.Li("Poor diet: Mediterranean"),
                        html.Li("BMI >30: Weekly glucose check")
                    ], style={'fontSize': '16px'})
                ], className="card p-4 m-3", style={'backgroundColor': '#fff3cd'})
            ], style={'maxWidth': '1000px', 'margin': 'auto'})
        ]),
        dcc.Tab(label='Demo Dataset', children=[
            html.Button('📋 Load Demo Data (20 rows)', id='load-data', n_clicks=0, className="btn btn-success btn-lg mb-4"),
            html.Div(id='data-table')
        ])
    ])
], style={'fontFamily': 'Arial, sans-serif', 'minHeight': '100vh'})

@callback(
    [
        Output('prediction-output', 'children'),
        Output('age', 'value'),
        Output('gender', 'value'),
        Output('bmi', 'value'),
        Output('physical_activity', 'value'),
        Output('diet_score', 'value'),
        Output('sleep_hours_per_day', 'value')
    ],
    Input('predict-btn', 'n_clicks'),
    [
        State('age', 'value'),
        State('gender', 'value'),
        State('bmi', 'value'),
        State('physical_activity', 'value'),
        State('diet_score', 'value'),
        State('sleep_hours_per_day', 'value')
    ]
)
def update_prediction(n_clicks, age, gender, bmi, phys_act, diet_score, sleep_hours_per_day):
    if n_clicks == 0:
        return "", None, None, None, None, None, None
    
    input_data = {
        'Age': age or 45,
        'gender': gender or 0,
        'bmi': bmi or 25.0,
        'physical_activity_minutes_per_week': {'High': 300, 'Medium': 150, 'Low': 50}[phys_act or 'Medium'],
        'diet_score': diet_score or 7.0,
        'sleep_hours_per_day': sleep_hours_per_day or 7,
        **{
            'family_history_diabetes': 0, 'hypertension_history': 0, 'cardiovascular_history': 0,
            'alcohol_consumption_per_week': 2, 'screen_time_hours_per_day': 5, 'waist_to_hip_ratio': 0.9,
            'systolic_bp': 120, 'diastolic_bp': 80, 'heart_rate': 72,
            'cholesterol_total': 200, 'hdl_cholesterol': 50, 'ldl_cholesterol': 120, 'triglycerides': 150,
            'ethnicity': 'White', 'education_level': 'Highschool', 'income_level': 'Middle', 'employment_status': 'Employed', 'smoking_status': 'Never'
        }
    }
    
    risk_result = predict_risk(input_data)
    cluster = get_cluster(input_data)
    
    probs = sorted(risk_result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    return html.Div([
        html.Hr(),
        html.H2(f"Risk Stage: {risk_result['risk_stage']}", style={'color': '#28a745'}),
        html.H3("Probabilities", style={'color': '#007bff'}),
        html.Table([
            html.Tr([html.Th("Stage"), html.Th("Probability")], style={'backgroundColor': '#dee2e6'}),
            *[html.Tr([html.Td(stage, style={'fontWeight': 'bold' if prob > 0.2 else 'normal'}), html.Td(f"{prob*100:.1f}%")]) for stage, prob in probs[:5]]
        ], style={'width': '100%', 'margin': '20px 0'}),
        html.H2(f"Segment: Cluster {cluster}", style={'color': '#dc3545'}),
        html.Div([
            html.P("Cluster 0: Low Risk ✓"),
            html.P("Cluster 1: Medium - Monitor"),
            html.P("Cluster 2: High - Immediate Action ⚠️")
        ], style={'backgroundColor': '#fff3cd', 'padding': '20px', 'borderRadius': '10px'}),
    ], style={'border': '2px solid #007bff', 'padding': '30px', 'borderRadius': '15px', 'boxShadow': '0 8px 16px rgba(0,0,0,0.1)'}), None, None, None, None, None, None

@callback(
    Output('data-table', 'children'),
    Input('load-data', 'n_clicks')
)
def load_demo(n):
    if n == 0:
        return ""
    import pandas as pd
    df = pd.read_parquet('data/Diabetes_and_Lifestyle_Dataset_.parquet').head(20)
    return dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        style_cell={'textAlign': 'left'},
        style_data_conditional=[
            {'if': {'filter_query': '{Diabetes_Stage} = 3'}, 'backgroundColor': '#f8d7da'}
        ]
    )

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 8050)))
