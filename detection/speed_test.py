import argparse 
import torch
import numpy as np

from model import get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--size', type=int)
    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()
    model = get_model(backbone_name=args.backbone, size=args.size)

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    dummy_input = torch.randn(1, 3, args.size, args.size, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(f"Mean time is {mean_syn}ms with std of {std_syn}")
    print(f"FPS is {1/(mean_syn/1000)}")

