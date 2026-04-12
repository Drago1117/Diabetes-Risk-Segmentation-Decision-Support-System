import dash
from dash import html, dcc
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Diabetes Risk Dashboard"),

    html.Label("Age"),
    dcc.Input(id='age', type='number'),

    html.Br(),

    html.Label("BMI"),
    dcc.Input(id='bmi', type='number'),

    html.Br(),

    html.Button("Predict", id='predict-btn'),

    html.Div(id='output')
])

@app.callback(
    Output('output', 'children'),
    Input('predict-btn', 'n_clicks'),
    Input('age', 'value'),
    Input('bmi', 'value')
)
def predict(n, age, bmi):
    if n is None:
        return ""
    
    # TEMP FAKE OUTPUT
    return f"Risk: High Risk (Age: {age}, BMI: {bmi})"

if __name__ == '__main__':
    app.run(debug=True)