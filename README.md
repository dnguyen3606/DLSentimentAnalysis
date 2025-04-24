# DLSentimentAnalysis
This is the project repository for a CS4644 project for music generation based on emotion labeled text.

To run, either use:
- CLI: `python main.py "text input of choice"`
- Streamlit: `streamlit run app.py`

If using CLI, navigate to output folder to find .mid and .wav outputs of music.

We use the GoEmotions dataset trained using our own LSTM architecture to label text with probabilities that the text conveys any of the 28 emotions. We then translate these probabilities into valence-arousal levels, which we then input into EMOPIA, which generates music based off of valence-arousal.

## F.A.Q.

<details>
  <summary>Q1. How do I run this?</summary>
  Install required large files:

  - [EMOPIA Transformer Checkpoint](https://drive.google.com/file/d/19Seq18b2JNzOamEQMG1uarKjj27HJkHu/view)
    - Unzip 'loss_25_params.pt' into '/models/' at the same depth as the LSTM .pth.
  - [EMOPIA Dictionary](https://drive.google.com/file/d/17dKUf33ZsDbHC5Z6rkQclge3ppDTVCMP/view)
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

  Note: If you are using Linux, you can sidestep this issue by using `apt-get install fluidsynth`
</details>
