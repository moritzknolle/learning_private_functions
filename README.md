
# Learning private functions from sensitive data

This project explores combining sparse Gaussian process approximations with differential privacy, to obtain reliable, private models.

![Alt text](figs/main_fig.png?raw=true "Private vs. non-private sparse GP approximations")

## Getting started 

Clone the project

```bash
  git clone https://github.com/moritzknolle/learning_private_functions.git
```

Go to the project directory

```bash
  cd learning_private_functions
```

Install dependencies

```bash
  conda env create -f environment.yml
```

Train your first differentially private Gaussian process

```bash
  python dp_svgp.py
```

