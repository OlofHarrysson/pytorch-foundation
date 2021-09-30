import sys


def find_best_learning_rate(trainer, model, dataloaders):
    lr_finder = trainer.tuner.lr_find(model, dataloaders.train)
    if lr_finder:
        fig = lr_finder.plot(suggest=True)
        fig.show()
        print(f"Suggested learning rate: {lr_finder.suggestion()}")
        input("Showing learning rate graph. Press Enter key to quit")
    sys.exit(0)


if __name__ == "__main__":
    from train import main

    main(find_best_lr=True)
