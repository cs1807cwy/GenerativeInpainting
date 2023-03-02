# main.py
from pytorch_lightning.cli import LightningCLI

# simple demo classes for your convenience
from Inpaint import *


def cli_main():
    cli = LightningCLI()
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block