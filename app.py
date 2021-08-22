from dash_html_components.H1 import H1
from learning.DataProvider import DataProvider
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from dash_html_components.Div import Div
from helpers import load_model, predict, load_dp
from learning.Model import Model


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)


app.layout = html.Div([
    html.H1('Sentiment analysis app'),

    html.Div([
        html.Div([
            html.Div([
            dcc.Textarea(
                id='text_box',
                placeholder='Please enter text to analyze...',
                spellCheck = True,
                style={'width': '49%', 'display': 'inline-block','height' : 200},
            ),
            html.Div([
                html.Button(
                    'Analys my text', id='submit-val', n_clicks=0
                    )
                ],style={'width': '49%', 'display': 'inline-block',
                'position': 'absolute', 'margin-left': '5px'})
            ])
        ]),
        html.H1(
            'Hi', id='semantic')
            ]),
        html.Div(
            html.Img(id = 'emojy',
            src = 'https://api.iconify.design/twemoji/check-mark-button.svg',
            style={'width': '100px'})
            
        ),
        html.Div('Produce By Or Levi')
])


@app.callback(Output('semantic', 'children'),
              Input('submit-val', 'n_clicks'),
              Input('text_box', 'value'))
def predict_semantic(btn1, text):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'submit-val' in changed_id:
        model = load_model()
        model_to_predict: Model = Model()
        model_to_predict.set_model(model=model)
        dp = load_dp()
        model_to_predict.set_dp(dp=dp)
        model_to_predict.word_embeding()
        p = predict(model_to_predict, text)
        return ('your text is: ' + p)
    else:
        return('Hi, this is a sentiment analysis app.')

@app.callback(Output('emojy', 'src'),
              Input('semantic', 'children'))
def predict_semantic(semantic_res):

    src= 'https://api.iconify.design/twemoji/check-mark-button.svg'
    
    if semantic_res == 'your text is: Negative':
        src = 'https://api.iconify.design/twemoji/angry-face.svg'
    
    if semantic_res == 'your text is: Neutral':
        src = 'https://api.iconify.design/twemoji/expressionless-face.svg'
    
    if semantic_res == 'your text is: Positive':
        src = 'https://api.iconify.design/twemoji/grinning-face-with-smiling-eyes.svg'
    
    return src




if __name__ == "__main__":
    app.run_server(debug=True)
