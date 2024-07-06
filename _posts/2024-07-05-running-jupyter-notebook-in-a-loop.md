## Running Jupyter Notebooks in a loop

If you've ever experimented with training your own machine learning models (e.g., using PyTorch or TensorFlow), then you know how random their results can occasionally be. In one run, you might achieve 90% accuracy, while in another, the model could get stuck in a local minimum or reach 95% accuracy. Training and testing a model only once might not yield the most optimal result. Additionally, there's another argument against doing a single run: when you're conducting research, you may want to statistically determine which architecture among those you've developed is the most efficient or stable (e.g., consistently giving results in the 90%-95% range instead of varying widely). This information could guide your next steps.

Obviously, training an enormous LLM model takes a huge amount of time, so rerunning it many times would not be feasible. You would need to employ a different strategy to arrive at the most optimal result (which I'm yet to discover), but this approach can be used for smaller models.

You could run your notebooks manually, remembering to increment the run number each time so that your code saves checkpoints for later analysis. However, if you noticed the word "manually," you already know what I'm getting at. Manually doing this many times is prone to errors and takes our precious time. I don't know about you, but I became a programmer to save people's (including my own) time, not waste it.

In this article I'll walk you through a Python helper script that automates multiple Jupyter Notebook runs.

---

### Requirements for the helper script

1. It should get information on how many times to run the notebooks, along with the notebook list.
2. It should be possible to stop and restart from the last checkpoint.

### The code

The first function in the helper script is this one:

```python
def get_runs_data(allowed_files: set[str]) -> dict[str, int]:
    files = os.listdir(".")
    run_data = defaultdict(lambda: 1)

    for file in files:
        if file not in allowed_files:
            continue

        file_path = os.path.join(".", file)

        if os.path.isdir(file_path):
            continue

        file_base_name = file.split('.')[0]
        run_file_name = f'{file_base_name}'
        run_number = get_run_number(run_file_name)
        run_data[file_path] = run_number

    return run_data
```

This function retrieves a list of files (Jupyter Notebook paths) and checks how many runs have been completed for each notebook. This implementation addresses the second requirement. If you stopped at 10 out of 20 runs, on subsequent triggers it will complete the remaining 10 runs, not all 20. It references the `get_run_number` function, but since it's very simple, I'll leave it without additional comments:

```python
def _get_runs_file_path(run_file: str) -> str:
    return os.path.join(os.path.expanduser('~'), f'.{run_file}')


def get_run_number(run_file: str) -> int:
    run_files_path = _get_runs_file_path(run_file)
    run_file = Path(run_files_path)

    run_file.parent.mkdir(exist_ok=True, parents=True)
    run_file.touch(exist_ok=True)

    text = run_file.read_text()

    if len(text) > 0:
        return int(text)

    return 1
```

There's also another helper function for checking if the process should terminate. It goes to the user directory and checks if the so-called `exit_file` exists. Obviously, you could just `ctrl+c` to stop it, but that would immediately halt the current training. Employing a graceful shutdown strategy is better in this case.

```python
def should_exit(exit_file: str) -> bool:
    path = os.path.join(os.path.expanduser('~'), f'.{exit_file}')
    return os.path.isfile(path)
```

### Gluing it all together

This is the main part of the script. The `load_dotenv()` call is here because, in my projects, I like to use `.env` files to store my configurations. You can also notice that I'm using the `tqdm` library for nice-looking progress bars. However, given this script is made for running in the console, it won't be as nice as it would be if run in a notebook (new progress bars will be produced at each step instead of updating the existing one). Most of the code is simple, except perhaps for the lines defining the `conda_activate_command` and `command` variables. The first one activates the conda environment of your project, and the second one uses the Jupyter CLI tool to run the notebooks. You may have noticed the setting of the `RUN_NUMBER` environment variable - it's done so that the notebook code knows what number to put on the checkpoint it saves. This could have been implemented differently, for example, after the notebook is run, this script could check the last checkpoint file created and increment its number, making the notebook completely independent of environment variables, but this approach was just shorter to write.

```python
load_dotenv()

RUN_TIMES = int(os.getenv("RUN_TIMES"))
EXIT_FILE = os.getenv("EXIT_FILE")
NOTEBOOKS = set(os.getenv("NOTEBOOKS").split(","))
runs_data = get_runs_data(NOTEBOOKS)
total_runs = len(runs_data) * RUN_TIMES

with tqdm(total=total_runs, desc="Processing Notebooks") as pbar:
    for notebook_path, run in runs_data.items():
        if run >= RUN_TIMES + 1:
            continue

        for counter in range(RUN_TIMES - run + 1):
            if should_exit(EXIT_FILE):
                print('Exit file encountered, aborting...')
                sys.exit(0)

            conda_activate_command = "conda activate bounding_box_detection_ham10000_torch && "
            command = (f"{conda_activate_command}jupyter nbconvert --execute --to notebook --inplace "
                       f"--ExecutePreprocessor.kernel_name=python3 {notebook_path}")
            os.environ["RUN_NUMBER"] = str(run)
            os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = str(1)

            subprocess.run(command, shell=True)
            save_run_data(notebook_path.split(os.sep)[-1], run + 1)
            pbar.update(1)

            run += 1
```

### Summary and next steps

I use this script when experimenting with neural networks, and I also used it while working on my engineering thesis. The next step I have in mind is to add functionality for using different machines. That way, different architectures could be run simultaneously, speeding up the process. If you immediately thought that some frameworks allow for distributed training, you're right. However, for small networks, the synchronization overhead would not be worth it, and I believe it would be more performance-efficient to run multiple notebooks in parallel.
