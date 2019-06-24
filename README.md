# DL-toolbox

Deep Learning Toolbox in PyTorch with Tensorboard with Tensorboard.

1. Train a model.

    ```shell
    python main.py -h
    usage: main.py [-h] [-lr LEARNING_RATE] [-b BATCH_SIZE] [-c] [-e EPOCHS]
               [--save-every SAVE_EVERY] [--print-every PRINT_EVERY]
               [--gpus GPUS] [--data DATA] [--model MODEL]

    optional arguments:
      -h, --help            show this help message and exit
      -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                            Assign learning rate, default 1e-4.
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            Assign batch size, default 64.
      -c, --continue        Continue training, default False.
      -e EPOCHS, --epochs EPOCHS
                            Epochs to train, default 20.
      --save-every SAVE_EVERY
                            Save the model every n epochs, default 5.
      --print-every PRINT_EVERY
                            Print the model every n steps, default 10.
      --gpus GPUS           Choose the total number of gpus to use.
      --data DATA           Set the data path.
      --model MODEL         Set the model path.
    ```

2. Generate a submit file.

    ```shell
    python submit.py
    usage: main.py [-h] [-b BATCH_SIZE] [--data DATA] [--model MODEL]
               [--gpus GPUS]

    optional arguments:
      -h, --help            show this help message and exit
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            ssign batch size, default 64.
      --data DATA           Set the data path.
      --model MODEL         Set the model path.
      --gpus GPUS           Choose the total number of gpus to use.
    ```

3. Model zoo.

    > Usage: ```python main.py --model MODEL_NAME```

    * Efficientnetb3
    * Efficientnetb5
    * Resnet152
    * SENet154

