# Easy Lightning: 🚀 Streamlining Deep Learning with PyTorch Lightning

**Easy Lightning** is a versatile Python library meticulously crafted to streamline the process of configuring and building AI-Deep learning models using the power of PyTorch Lightning and PyTorch models. With its unique configuration-based approach, Easy Lightning empowers developers and data scientists to effortlessly harness the full potential of deep learning.

## 🌟 Key Features

1. **User-Friendly Configuration:** Easy Lightning revolutionizes the way you work with deep learning by providing a seamless configuration-based interface. All your settings can be easily defined in YAML files, making it painless to tweak and fine-tune your experiments.

2. **Three Essential Utilities:**
   - 📊 **Data:** Easy Lightning simplifies data handling, allowing you to load and preprocess data effortlessly.
   - 📝 **Experiments:** Define, track, and save experiments with unique IDs to prevent duplication. Keep your work organized and accessible.
   - ⚙️ **Torch Integration:** Seamlessly integrate PyTorch models, train, test, and save your models with minimal effort. Easy Lightning handles the heavy lifting, so you can focus on innovation.

## 🚀 Why Choose Easy Lightning?

- **Efficiency:** Say goodbye to repetitive and time-consuming setup tasks. Easy Lightning automates the tedious parts of deep learning, giving you more time for experimentation and innovation.

- **Flexibility:** Whether you're a seasoned deep learning practitioner or just getting started, Easy Lightning adapts to your needs. Its configuration-based approach makes it accessible to all skill levels.

- 🛡️ **Error Prevention:** With experiment IDs and organized project management, Easy Lightning ensures that your work remains clean and error-free, even as your projects scale.

## 📁 Important Files and Folders in the Project

Below is an outline of key files and folders you'll find in this project, along with their purposes:

### Files

1. **setup.py**
    - Setup script for Python's setuptools. Specifies package metadata, dependencies, and other distribution essentials.

2. **requirements.txt**
    - Lists required Python packages and their versions. Ensures that necessary dependencies are installed.
  
    - Currently includes:
        - PyTorch: Popular deep learning library.
        - PyTorch Lightning: Lightweight PyTorch wrapper.

3. **README.md**
    - Provides an overview of the project and usage instructions.

4. **.gitignore**
    - Specifies files and folders that should be ignored by Git, ensuring that unnecessary or sensitive data is not included in the version control.

### Folders

1. **easy_data**
    - Contains utilities for data loading, file management, data and data structure management, data splitting, and statistics.

2. **easy_exp**
    - Manages experiments by defining unique IDs based on their configuration.
    - Allows for hashing of each ID to check for previously conducted experiments.
    - Excludes GPU/CPU usage and training modes from the experiment ID.
    - Saves experiments in a specific file along with their relative configuration.
    - Includes methods for parsing YAML configs and handles special characters used in them (e.g., through `var.py`).

3. **easy_torch**
    - Includes functions for metrics, loading models, and creating trainers in PyTorch Lightning.
    - Defines steps, loss, optimizer, and other parameters to use.
    - Sets callbacks and dataloaders.
    - Manages TorchVision models, allowing you to load them and modify their internal modules if necessary.
    - Also includes utilities for training and testing the model, as well as saving and reading logs.

4. **cfg**
    - Contains demo configurations used in testing phase of this repo.

5. **ntb**
    - Houses three notebooks for testing the utilities offered by the three utility folders.
    - Also includes a notebook with information about the objectives and logic used in the implementation.

By understanding the role of each file and folder, you'll be better equipped to navigate and work on the project.

## 🧑‍🚀 Authors

- **Federico Siciliano**: Researcher in Data Science at the University of La Sapienza of Rome.
- **Federico Carmignani**: Master's Thesis Student in Engineering in Computer Science at the University of La Sapienza of Rome and Software Engineer at Sytel Reply.


Start your deep learning journey with Easy Lightning today and experience a new level of simplicity and efficiency in creating and configuring AI-Deep learning notebooks.
