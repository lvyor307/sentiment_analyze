import dash
import dash_html_components as html
import dash_core_components as dcc


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = html.Div(
    [
       dcc.Textarea(id = 'text_box',
                    placeholder='Enter a value...',
                    style={'width': '100%'}
                    )  
    ]
)




if __name__ == "__main__":
    app.run_server(debug=True)
