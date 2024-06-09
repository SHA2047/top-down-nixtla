```mermaid
graph TD
    A[Start] --> B[Initialize Streamlit App]
    B --> C[Upload Input File]
    C --> D{Is a file uploaded?}
    D -- No --> E[Wait for file upload]
    D -- Yes --> F[Read CSV and Display Preview]
    F --> G[Specify Dependent Variable]
    G --> H[Select Univariate Models]
    H --> I[Run Forecasting Comparison Button]
    I --> J{Is button clicked?}
    J -- No --> K[Wait for button click]
    J -- Yes --> L{Are models selected?}
    L -- No --> M[Display error message]
    L -- Yes --> N[Run Forecasting Comparison]
    N --> O[Display Results]
    O --> P[End]



### Detailed Steps

```markdown
1. **Start**
   - Streamlit: `st.title("Univariate Time Series Forecasting Comparison")`

2. **Upload Input File**
   - Streamlit: `uploaded_file = st.file_uploader("Upload CSV", type=["csv"])`
   - **Decision:** `if uploaded_file is not None`
     - **Yes:**
       - `data = pd.read_csv(uploaded_file)`
       - `st.write("Data preview:")`
       - `st.dataframe(data.head())`
     - **No:** (No action, wait for file upload)

3. **Specify Dependent Variable**
   - Streamlit: `dependent_var = st.selectbox("Select the dependent variable:", data.columns)`

4. **Select Univariate Models**
   - Streamlit: `model_options = [...]`
   - Streamlit: `selected_models = st.multiselect("Select univariate models:", model_options, default=model_options)`

5. **Run Forecasting Comparison Button**
   - Streamlit: `if st.button("Run Forecasting Comparison"):`
   - **Decision:** `if not selected_models`
     - **Yes:** `st.error("Please select at least one model.")`
     - **No:**
       - **Try Block:**
         - `results, best_model_name, future_forecast = run_forecasting_comparison(data, dependent_var, selected_models)`
         - `st.write("Model Comparison Results:")`
         - `st.write(results)`
         - `st.write(f"Best Model: {best_model_name}")`
         - `st.write("Future Forecast:")`
         - `st.write(future_forecast)`
       - **Except Block:**
         - `st.error(f"An error occurred: {e}")`

6. **End**

