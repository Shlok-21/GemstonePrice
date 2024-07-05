import app_streamlit as st
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Define the main function for the Streamlit app
def main():
    st.title('Gemstone Price Prediction')
     # Define a sidebar for the data dictionary
    st.sidebar.title('Data Dictionary')
    st.sidebar.markdown("""
    **Carat**: Carat weight of the cubic zirconia.

    **Cut**: Describe the cut quality of the cubic zirconia. Quality is in increasing order from Fair to Ideal.

    **Color**: Color of the cubic zirconia. D is the best and J is the worst.

    **Clarity**: Clarity refers to the absence of inclusions and blemishes in the cubic zirconia. 
    Clarity grades range from IF (Internally Flawless) to I1 (Included).

    **Depth**: Height of the cubic zirconia, measured from the Culet to the table, divided by its average girdle diameter.

    **Table**: Width of the cubic zirconia's table expressed as a percentage of its average diameter.

    **Price**: Price of the cubic zirconia.

    **X, Y, Z**: Dimensions of the cubic zirconia in millimeters (mm).
    """)


    # Collect user input
    carat = st.text_input('Carat', placeholder='Enter carat (e.g., 0.5 - 5)')
    cut = st.selectbox('Cut', ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair'], help='Select the cut grade of the gemstone')
    color = st.selectbox('Color', ['D', 'E', 'F', 'G', 'H', 'I', 'J'], help='Select the color grade of the gemstone')
    clarity = st.selectbox('Clarity', ['SI1', 'IF', 'VVS2', 'VS1', 'VVS1', 'VS2', 'SI2', 'I1'], help='Select the clarity grade of the gemstone')
    depth = st.text_input('Depth', placeholder='Enter depth (e.g., 50 - 80)')
    table = st.text_input('Table', placeholder='Enter table size (e.g., 50 - 70)')
    x = st.text_input('X', placeholder='Enter dimension X (e.g., 3 - 10)')
    y = st.text_input('Y', placeholder='Enter dimension Y (e.g., 3 - 10)')
    z = st.text_input('Z', placeholder='Enter dimension Z (e.g., 1 - 6)')
    

    # When the predict button is clicked
    if st.button('Predict gem price'):
        data = CustomData(
            carat=carat,
            cut=cut,
            color=color,
            clarity=clarity,
            depth=depth,
            table=table,
            x=x,
            y=y,
            z=z,
        )
        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.success(f'Predicted gem price: {results[0]} dollars.')  # Display prediction result

# Entry point of the Streamlit app
if __name__ == '__main__':
    main()
