# Simple RNN Music Generation Project

This project demonstrates how to build and train a Recurrent Neural Network (RNN) using TensorFlow/Keras to generate new, simple monophonic melodies from existing MIDI music. It covers the end-to-end process from data preparation to music synthesis.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [How It Works](#how-it-works)
4. [Setup](#setup)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Obtaining MIDI Data](#obtaining-midi-data)
5. [Usage](#usage)
    - [Configuration](#configuration)
    - [Running the Script](#running-the-script)
    - [Listening to Generated Music](#listening-to-generated-music)
6. [Project Structure](#project-structure)
7. [Future Enhancements](#future-enhancements)
8. [License](#license)
9. [Acknowledgments](#acknowledgments)

---

## 1. Project Overview

This project implements a basic deep learning pipeline for symbolic music generation. It takes a collection of MIDI files, processes them into a sequence of numerical events (pitch and duration pairs), trains an LSTM-based RNN to learn the patterns in these sequences, and then uses the trained model to generate novel musical sequences, which are finally converted back into a playable MIDI file.

---

## 2. Features

- **MIDI Parsing:** Reads and extracts note information from standard MIDI files using `pretty_midi`.
- **Data Preprocessing:** Converts raw note data into a quantized sequence of (pitch, duration) events.
- **Sequence Creation:** Prepares input-output pairs for RNN training using a sliding window approach.
- **RNN Model:** Builds and compiles a simple LSTM network with an embedding layer using TensorFlow/Keras.
- **Model Training:** Trains the RNN to predict the next musical event in a sequence.
- **Model Checkpointing:** Saves the best performing model weights during training.
- **Music Generation:** Generates new musical sequences by iteratively predicting notes based on a seed sequence and learned probabilities, with adjustable "temperature" for creativity.
- **MIDI Output:** Converts the generated numerical sequences back into a playable MIDI file.

---

## 3. How It Works

The process can be broken down into these main steps:

1. **Data Collection (MIDI Files):** The project starts with `.mid` files, which contain symbolic musical information (like note pitches, durations, and timing) rather than raw audio.
2. **Data Preprocessing:**
    - Each note from the MIDI files is extracted.
    - These notes are simplified into (pitch, duration) pairs.
    - Durations are quantized to discrete values (e.g., quarter notes, eighth notes) to simplify the learning task.
3. **Sequence Creation:**
    - All unique (pitch, duration) pairs form a vocabulary, and each is mapped to a unique integer ID.
    - The entire musical piece is transformed into a sequence of these integer IDs.
    - A "sliding window" approach is used to create input-output training examples. For example, if `SEQUENCE_LENGTH` is 50, the model is trained to predict the 51st event given the previous 50 events.
4. **Model Building (RNN/LSTM):**
    - A Keras Sequential model is defined.
    - An Embedding layer converts the integer IDs into dense vector representations.
    - Two LSTM layers learn long-term dependencies and patterns within the musical sequences.
    - A Dropout layer helps prevent overfitting.
    - A final Dense layer with softmax activation outputs a probability distribution over the entire vocabulary, indicating the likelihood of each possible next event.
5. **Model Training:** The model is trained on the prepared input-output sequences. It adjusts its internal weights to minimize the difference between its predictions and the actual next events, gradually learning the "grammar" of the music.
6. **Music Generation:**
    - A short "seed" sequence (taken from the training data) is fed to the trained model.
    - The model predicts the probabilities for the next event.
    - Instead of picking the most probable event, an event is sampled from this probability distribution. A temperature parameter can be adjusted to control the randomness (creativity vs. coherence).
    - The sampled event is appended to the current sequence, and the oldest event is removed, maintaining the `SEQUENCE_LENGTH`.
    - This process repeats iteratively to generate a new melody of a desired length.
7. **MIDI Reconstruction & Audio Output:** The generated sequence of integer IDs is converted back into (pitch, duration) tuples, which are then used to construct a new `pretty_midi` object. This object is saved as a `.mid` file, which can be played back by any MIDI player or converted to an audio format (like WAV or MP3).

---

## 4. Setup

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone the repository (if applicable) or create a new project directory.
2. Navigate to your project directory in the terminal/command prompt.
3. Install the required Python libraries:

    ```bash
    pip install pretty_midi numpy tensorflow
    # pretty_midi might have an old Keras dependency, this can sometimes resolve issues:
    pip install 'keras<2.0.0'
    ```

4. **Optional:** For MIDI to WAV/MP3 conversion on your local machine (outside of Python):

    - **Linux/macOS:** Install fluidsynth or timidity.

        ```bash
        # For Ubuntu/Debian:
        sudo apt-get update
        sudo apt-get install fluidsynth timidity

        # For macOS (using Homebrew):
        brew install fluidsynth timidity
        ```

    - You might also need a SoundFont (`.sf2` file) for fluidsynth to synthesize sounds. A common one is `FluidR3_GM.sf2`, often found in `/usr/share/sounds/sf2/`.

### Obtaining MIDI Data

For this project, you need a MIDI file to train the model.

1. **Download a simple monophonic MIDI file:**  
   Search online for "simple monophonic MIDI file download" or "public domain MIDI melodies".
2. **Place the MIDI file:**  
   Save your chosen MIDI file (e.g., `my_melody.mid`) in the same directory as your `midi_gen_project.py` script.
3. **Update MIDI_FILE_PATH:**  
   Ensure the `MIDI_FILE_PATH` variable in the script points to your MIDI file.

---

## 5. Usage

### Configuration

Open the `midi_gen_project.py` file and adjust the configuration variables at the top of the script as needed:

```python
# --- Configuration ---
MIDI_FILE_PATH = 'simple_melody.mid' # Path to your input MIDI file
SEQUENCE_LENGTH = 50                 # Number of notes/events in the input sequence for the RNN
BATCH_SIZE = 64                      # Number of samples per gradient update during training
EPOCHS = 50                          # Number of times the model iterates over the entire dataset
LSTM_UNITS = 256                     # Number of units (neurons) in the LSTM layers
EMBEDDING_DIM = 100                  # Dimension of the embedding vector for each event
MODEL_WEIGHTS_PATH = 'music_model_weights.h5' # File to save/load trained model weights
GENERATED_MIDI_PATH = 'generated_melody.mid'  # Output path for the generated MIDI file
GENERATION_LENGTH = 100              # Number of new notes/events to generate
```

- **MIDI_FILE_PATH:** Crucial! Set this to the path of the MIDI file you downloaded.
- **SEQUENCE_LENGTH:** Controls how much "memory" the RNN has. Larger values mean more context, but require more data and computation.
- **EPOCHS:** Increase for more training, decrease if overfitting or training is too slow.
- **GENERATION_LENGTH:** How long you want your generated melody to be.

### Running the Script

1. **Save the script:** Ensure all the provided code snippets are combined into a single Python file (e.g., `midi_gen_project.py`).
2. **Execute from terminal:**

    ```bash
    python midi_gen_project.py
    ```

The script will perform the following steps:

- Load and inspect your input MIDI file.
- Process the notes into a numerical sequence.
- Create input-output training pairs.
- Build the RNN model.
- If `music_model_weights.h5` does not exist: The model will train for the specified number of epochs, saving the best weights. This step can take some time depending on your dataset size and hardware.
- If `music_model_weights.h5` exists: The script will load the pre-trained weights, skipping the training phase.
- Generate a new melody using the trained model.
- Save the generated melody as `generated_melody.mid` in the same directory.

### Listening to Generated Music

After the script completes, a new MIDI file named `generated_melody.mid` (or your configured `GENERATED_MIDI_PATH`) will be created.

You can listen to it by:

- **Uploading to an online MIDI player:** Search for "online MIDI player" and upload your generated file.
- **Using a desktop media player:** Most modern media players (like VLC) can play MIDI files, though they might use a default soundfont.
- **Converting to WAV/MP3 (requires fluidsynth or timidity):**

    ```bash
    # Using timidity (simpler):
    timidity generated_melody.mid -Ow -o generated_melody.wav

    # Using fluidsynth (requires a soundfont, e.g., FluidR3_GM.sf2):
    fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 generated_melody.mid -F generated_melody.wav -r 44100
    ```

    Then, play the generated `.wav` file.

---

## 6. Project Structure

```
.
├── midi_gen_project.py
├── simple_melody.mid         # Your input MIDI file (example name)
├── music_model_weights.h5    # Generated after first training run
└── generated_melody.mid      # Generated after music generation step
```

- **midi_gen_project.py:** Contains all the Python code for data loading, preprocessing, model building, training, and music generation.
- **simple_melody.mid:** The MIDI file used as training data.
- **music_model_weights.h5:** The file where the trained neural network's weights are saved.
- **generated_melody.mid:** The MIDI file containing the newly generated music.

---

## 7. Future Enhancements

This project provides a solid foundation. Here are some ideas for further exploration:

- **Larger/Diverse Datasets:** Train on more extensive and varied MIDI datasets (e.g., the full MAESTRO dataset or Lakh MIDI Dataset) for more complex and diverse musical output.
- **Polyphonic Music Generation:** Extend the model to generate multiple notes simultaneously (chords) or multiple instrument tracks. This would require a more complex output representation (e.g., multi-hot encoding or predicting multiple (pitch, duration) pairs per timestep).
- **Predicting Velocity/Instrument:** Currently, velocity is fixed. The model could be extended to predict note velocity and even instrument changes.
- **Attention Mechanisms:** Implement Transformer-based models or attention mechanisms within RNNs for better long-range coherence in generated music.
- **Conditional Generation:** Allow the user to specify conditions for generation (e.g., genre, mood, key, or a starting motif).
- **Web Interface:** Create a simple web application (using Flask or Django) to allow users to interact with the model and generate music through a browser.
- **Evaluation Metrics:** Implement quantitative metrics to evaluate the musicality of generated pieces (e.g., pitch range, note density, rhythmic complexity).
- **Hyperparameter Tuning:** Systematically experiment with `SEQUENCE_LENGTH`, `LSTM_UNITS`, `EMBEDDING_DIM`, `BATCH_SIZE`, and `EPOCHS` to optimize model performance.
- **Temperature Control:** Add a user input for the temperature parameter during generation to easily experiment with creativity.

---

## 8. License

This project is open-source and available under the MIT License.

---

## 9. Acknowledgments

- Inspired by the MIT Introduction to Deep Learning (6.S191) course.
- Utilizes the pretty_midi library for MIDI processing.
- Built with TensorFlow and Keras.

