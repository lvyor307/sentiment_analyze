import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from dash_html_components.Div import Div
from helpers import load_model, predict
from learning.Model import Model

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)

app.layout = html.Div([
    html.Div([
        html.H1('Hi', id='semantic')
    ]),
    html.Div([
        dcc.Textarea(
            id='text_box',
            placeholder='Enter a value...',
            value=' '
        )
    ]),
    html.Div([
        html.Button(
            'Submit', id='submit-val', n_clicks=0
        )
    ])
])


@app.callback(Output('semantic', 'children'),
              Input('submit-val', 'n_clicks'),
              Input('text_box', 'value'))
def predict_semantic(btn1, text):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-val' in changed_id:
        model = load_model()
        model_to_predict: Model = Model()
        model_to_predict.add_model_to_model(model=model)
        p = predict(model, text)
        return p
    else:
        return('Hi')


if __name__ == "__main__":
    app.run_server(debug=True)
