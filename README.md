# DLSentimentAnalysis

## F.A.Q.

<details>
  <summary>Q1. How do I run this?</summary>
  Install required large files:

  - [EMOPIA Transformer Checkpoint]()
    - Unzip 'loss_25_params.pt' into '/models/' at the same depth as the LSTM .pth.
  - EMOPIA Dictionary
    - Run `gdown --id 17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP`
    - Unzip files into '/data/emopia/co-representation'
   
  Install required packages; there is no 'requirements.txt'. Some unexpected packages are:
  - `pip install gdown`
  - `pip install pytorch-fast-transformers`
  - `pip install numpy==1.26.4` (if your current version is >v2.0.0
</details>

<details>
  <summary>Q2. I am running into 'fluidsynth' import errors? </summary>
  The currently recognized pypi package for `pip install fluidsynth` is **not** the correct fluidsynth package.

  For some reason it is an abandoned v0.2 package from 2012. While `pip install pyfluidsynth` is correct and updated, it does not create the bin file that fluidsynth requires.

  Instead, use Chocolatey ([install instructions here](https://chocolatey.org/install)) and run `choco install fluidsynth`. This is according to fluidsynth's actual installation instructions found on [their website](https://www.fluidsynth.org/download/)
</details>

<details>
    <summary> Q3. How do I Run Streamlit based Music Generator Chatbot</summary>

    A simple Streamlit app that shows how to build a chatbot using OpenAI's GPT-3.5.

    [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

    1. Install the requirements

    ```
    $ pip install -r requirements.txt
    ```

    2. Run the app

    ```
    $ streamlit run streamlit_app.py
    ```

</details>