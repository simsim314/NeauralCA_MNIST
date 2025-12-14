# Neural “CA” Digit Refiner

This project runs a **50-step neural cellular-automaton-like process** on a 28×28 image of a handwritten digit. In the end detecting the handwritten digit using purely local (r = 4) information processing of an image. 

Try the live demo here: [https://simsim314.github.io/NeauralCA_MNIST/](https://simsim314.github.io/NeauralCA_MNIST/). When you open it, wait about **30 seconds** for the ONNX model to finish loading in the browser before pressing **Run**. You should see `Ready` in the Animation + Verdict panel. 

## Core idea

- **State = image**  
  There is **no separate hidden vector/state** stored anywhere. All information is carried forward through the **image outputs** produced at each step.

- **Local-only processing**  
  Each step is computed using only **local neighborhoods** (a small convolution kernel). No global attention, no fully-connected global features during the CA update.

- **50 different rules (one per step)**  
  Unlike a classic CA where the same rule repeats, this model has **50 distinct update rules**:
  - step 1 uses rule 1
  - step 2 uses rule 2
  - …
  - step 50 uses rule 50

  Each “rule” is a small conv block that outputs a new 28×28 image.

## How a step works

At step `t`, the model takes:
- the original input image
- plus the images produced by steps `1..t-1`

It **concatenates them as channels** and applies a small convolutional block to produce the next image:

- input channels at step `t`: `1 + (t-1)`
- output: one new image `y_t` (28×28), squashed to `[0,1]` by a sigmoid

This means the only “memory” is the stack of images already produced.

## Why it’s CA-like

It matches the CA spirit because:
- updates are **spatially local**
- the system evolves by repeatedly producing a new grid (image)
- the grid itself is the “state”

It differs from classic CA because:
- rules are **learned neural filters**, not discrete logic
- rules are **step-specific** (50 different rules)

## Output / “verdict”

After 50 steps, the final image (step 50) is treated as the refined digit output.

A simple classifier then scores which digit it resembles (e.g., template similarity), and reports:
- predicted digit
- confidence score

## What the demo shows

- input image (your drawing → cropped/squared → downscaled to 28×28)
- all 50 intermediate images
- an animation through the 50-step evolution
- the final predicted digit + confidence

## To Train

run `prepare_data.py` then `train_CA.py`. 
to animate locally run `animate_CA.py`.
To export model to onnx use `tools/export_onnx.py`.

