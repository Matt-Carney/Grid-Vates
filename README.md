# Grid Vātēs - A Dynamic Line Rating Educational Sandbox

## Overview
Welcome to **Grid Vātēs**! This project was developed as part of Georgia Tech CS6460 - Educational Technology. [Grid Vātēs](https://grid-vates.streamlit.app/) is an interactive web application designed to educate users about Dynamic Line Ratings (DLR) in transmission lines and their dependency on environmental conditions. The assocaited project paper can be found [here](Grid_Vates.pdf).

## Abstract
Transmission line load limits are largely determined by environmental conditions, with factors such as minimum clearance distance and maximum thermal thresholds being influenced by ambient temperature, wind speed, wind direction, and solar irradiance. Dynamic line ratings (DLR) present an beneficial approach to grid operation, optimizing load capacity while maintaining safety through the use of weather forecasts and real-time monitoring. Improving the accuracy of DLR forecasts has been the subject of extensive research, employing various techniques like time series modeling, deep learning, and specialty software. However, a gap remains in understanding and building intuition on the relationship between weather forecasts, their uncertainty, and their impact on dynamic line ratings. To address this gap, Grid Vātēs - A Dynamic Line Rating Experimental Sandbox was developed,  leveraging three main learning components: 1.) The benefits of DLR, 2.) The impact of uncertainty in probabilistic weather forecasts on DLR forecasts, and 3.) A scenario analysis knowledge check. Grid Vātēs offers a novel and interactive approach to understand the relationship between weather forecasts and their impact on dynamic line ratings

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/grid-vates.git
    cd grid-vates
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage
Access the web application at [Grid Vātēs](https://grid-vates.streamlit.app/). Navigate through the tabs on the left pane to explore the features of the sandbox.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.