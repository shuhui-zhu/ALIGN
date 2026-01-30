# Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents
## Initialization
1. Create a virtual environment by `conda create -n gossip python=3.13.1`
2. Activate this environment by `conda activate gossip`
3. Setup your environment by running `pip install -r requirements.txt`
4. Log wandb by `wandb login`
## Usage
1. Save all ur api keys in `~/.bashrc` or `~/.zshrc`, then execute `bash ~/.bashrc` or `bash ~/.zshrc` in the terminal.
2. Specify your hyperparameters (which game, llms, api, etc) in `conf/`. The configuration will automatically be passed to `main.py`. 
    - Modify `conf/config.yaml` to change the default experiment (game), llm model, and api.
    - Modify `conf/experiment/*.yaml` to change the environment/game settings.
3. Run the simulation via `python main.py`. Set WANDB_MODE to "disabled" in `main.py` if you don't want to log to wandb.
