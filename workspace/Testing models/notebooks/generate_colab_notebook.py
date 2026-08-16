
import json
import os

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def create_notebook():
    # Helper to remove local imports since code is in cells
    def remove_local_import(code):
        lines = code.splitlines(keepends=True)
        return [l for l in lines if "from genetic_env import" not in l and "import genetic_env" not in l]

    # Read source files
    env_code = read_file('genetic_env.py').splitlines(keepends=True)
    train_code = remove_local_import(read_file('ppo_genetic.py'))
    train_code = [l.replace('progress_bar=True', 'progress_bar=False') for l in train_code]
    
    recurrent_code = remove_local_import(read_file('recurrent_ppo.py'))
    recurrent_code = [l.replace('progress_bar=True', 'progress_bar=False') for l in recurrent_code]
    
    eval_code = remove_local_import(read_file('evaluate_agent.py'))

    # Notebook Structure
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# PPO Genetic Photobioreactor - Colab Training\n",
                    "This notebook runs the IBM (Individual-Based Model) simulation for algal growth optimization using PPO."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install Dependencies\n",
                    "!pip install stable-baselines3[extra] shimmy gymnasium"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Define Environment (`genetic_env.py`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": env_code
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Train Agent (`ppo_genetic.py`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": train_code
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Train Recurrent Agent (`recurrent_ppo.py`)\n",
                    "Time-aware agent using LSTM."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": recurrent_code
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Download Models\n",
                    "Download the trained model and normalization statistics immediately after training."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "try:\n",
                    "    from google.colab import files\n",
                    "    import os\n",
                    "\n",
                    "    if os.path.exists('ppo_genetic_ibm.zip'):\n",
                    "        files.download('ppo_genetic_ibm.zip')\n",
                    "\n",
                    "    if os.path.exists('recurrent_ppo_genetic_ibm.zip'):\n",
                    "        files.download('recurrent_ppo_genetic_ibm.zip')\n",
                    "\n",
                    "    if os.path.exists('vec_normalize.pkl'):\n",
                    "        files.download('vec_normalize.pkl')\n",
                    "\n",
                    "    if os.path.exists('recurrent_vec_normalize.pkl'):\n",
                    "        files.download('recurrent_vec_normalize.pkl')\n",
                    "except ImportError:\n",
                    "    print('Not running in Google Colab, skipping download.')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Evaluate Agent (`evaluate_agent.py`)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": eval_code
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    with open('colab_training.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("Notebook 'colab_training.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
