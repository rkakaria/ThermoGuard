import gradio as gr
import pandas as pd
import pickle
import sklearn

if gr.NO_RELOAD:
    model = pickle.load(open('./model/model.sav', 'rb'))
    columns = ['Vibration_X', 'Vibration_Y', 'Vibration_Z', 'Acoustic_Level', 'Temperature']

def fault_predictor(v_x, v_y, v_z, al, temp):
    x = pd.DataFrame([[v_x, v_y, v_z, al, temp]], columns = columns)
    result = str(model.predict(x))[2:-2]
    return result

demo = gr.Interface(
    fn = fault_predictor,
    inputs=[gr.Textbox(label = 'Vibration - X (Normalized)', info = 'Between 0 - 1.3'), 
        gr.Textbox(label = 'Vibration - Y (Normalized)', info = 'Between 0 - 1.3'), 
        gr.Textbox(label = 'Vibration - Z (Normalized)', info = 'Between 0 - 1.3'), 
        gr.Textbox(label = 'Acoustic Level - dB', info = 'Between 10 - 110'), 
        gr.Textbox(label = 'Temperature - ºC)', info = 'Between 50 - 110')], # units are extrapolated
    outputs=[gr.Textbox(label = 'Machine Status')],
    title = "Industrial Equipment Fault Prediction",
    examples = "examples",
    description = "Predict whether your rotating industrial equipment is experiencing a fault.",
    live = True,
    api_name="predict"
)

demo.launch()